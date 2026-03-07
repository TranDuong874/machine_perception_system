#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <jsoncpp/json/json.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ImuTypes.h"
#include "System.h"

namespace {

constexpr uint32_t kMaxMessageBytes = 64U * 1024U * 1024U;
std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running = false;
}

bool read_exact(int fd, void* buffer, size_t size) {
    auto* out = static_cast<unsigned char*>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t read_size = ::recv(fd, out + offset, size - offset, 0);
        if (read_size == 0) {
            return false;
        }
        if (read_size < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        offset += static_cast<size_t>(read_size);
    }
    return true;
}

bool write_exact(int fd, const void* buffer, size_t size) {
    const auto* in = static_cast<const unsigned char*>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::send(fd, in + offset, size - offset, 0);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        offset += static_cast<size_t>(written);
    }
    return true;
}

bool recv_framed_message(int fd, std::string* out_message) {
    uint32_t network_size = 0;
    if (!read_exact(fd, &network_size, sizeof(network_size))) {
        return false;
    }
    const uint32_t message_size = ntohl(network_size);
    if (message_size == 0 || message_size > kMaxMessageBytes) {
        return false;
    }
    std::string buffer(message_size, '\0');
    if (!read_exact(fd, &buffer[0], buffer.size())) {
        return false;
    }
    *out_message = std::move(buffer);
    return true;
}

bool send_framed_message(int fd, const std::string& message) {
    if (message.empty() || message.size() > kMaxMessageBytes) {
        return false;
    }
    const uint32_t network_size = htonl(static_cast<uint32_t>(message.size()));
    if (!write_exact(fd, &network_size, sizeof(network_size))) {
        return false;
    }
    return write_exact(fd, message.data(), message.size());
}

// Adapted from public-domain style base64 snippets.
const std::string kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::vector<unsigned char> base64_decode(const std::string& input) {
    std::vector<int> table(256, -1);
    for (size_t i = 0; i < kBase64Alphabet.size(); ++i) {
        table[static_cast<unsigned char>(kBase64Alphabet[i])] = static_cast<int>(i);
    }
    std::vector<unsigned char> decoded;
    int val = 0;
    int valb = -8;
    for (unsigned char c : input) {
        if (c == '=') {
            break;
        }
        if (table[c] == -1) {
            continue;
        }
        val = (val << 6) + table[c];
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

std::string tracking_state_to_string(int state) {
    switch (state) {
        case -1:
            return "SYSTEM_NOT_READY";
        case 0:
            return "NO_IMAGES_YET";
        case 1:
            return "NOT_INITIALIZED";
        case 2:
            return "TRACKING";
        case 3:
            return "RECENTLY_LOST";
        case 4:
            return "LOST";
        case 5:
            return "TRACKING_KLT";
        default:
            return "UNKNOWN";
    }
}

bool is_finite_se3(const Sophus::SE3f& pose) {
    const auto translation = pose.translation();
    const auto quaternion = pose.unit_quaternion();
    return std::isfinite(translation.x()) && std::isfinite(translation.y()) &&
           std::isfinite(translation.z()) && std::isfinite(quaternion.w()) &&
           std::isfinite(quaternion.x()) && std::isfinite(quaternion.y()) &&
           std::isfinite(quaternion.z());
}

void fill_pose(Json::Value* out, const std::string& translation_key,
               const std::string& quaternion_key, const Sophus::SE3f& pose) {
    Json::Value translation(Json::arrayValue);
    translation.append(pose.translation().x());
    translation.append(pose.translation().y());
    translation.append(pose.translation().z());
    (*out)[translation_key] = translation;

    const auto q = pose.unit_quaternion();
    Json::Value quaternion(Json::arrayValue);
    quaternion.append(q.w());
    quaternion.append(q.x());
    quaternion.append(q.y());
    quaternion.append(q.z());
    (*out)[quaternion_key] = quaternion;
}

Json::Value make_error_response(const std::string& tracking_state,
                                const std::string& error_message,
                                int64_t timestamp_ns = 0) {
    Json::Value response(Json::objectValue);
    response["type"] = "orb_response";
    response["timestamp_ns"] = Json::Int64(timestamp_ns);
    response["tracking_state"] = tracking_state;
    response["camera_translation_xyz"] = Json::Value(Json::nullValue);
    response["camera_quaternion_wxyz"] = Json::Value(Json::nullValue);
    response["body_translation_xyz"] = Json::Value(Json::nullValue);
    response["body_quaternion_wxyz"] = Json::Value(Json::nullValue);
    response["tracking_image_jpeg_b64"] = Json::Value(Json::nullValue);
    response["error"] = error_message;
    return response;
}

std::string to_json_string(const Json::Value& value) {
    Json::StreamWriterBuilder writer_builder;
    writer_builder["indentation"] = "";
    return Json::writeString(writer_builder, value);
}

bool parse_json(const std::string& input, Json::Value* out, std::string* errors) {
    Json::CharReaderBuilder reader_builder;
    std::unique_ptr<Json::CharReader> reader(reader_builder.newCharReader());
    return reader->parse(input.data(), input.data() + input.size(), out, errors);
}

class OrbNativeService {
   public:
    OrbNativeService(const std::string& voc_file, const std::string& settings_file, bool use_viewer)
        : slam_(voc_file, settings_file, ORB_SLAM3::System::IMU_MONOCULAR, use_viewer, 0, "") {}

    ~OrbNativeService() {
        try {
            slam_.Shutdown();
        } catch (...) {
        }
    }

    Json::Value process(const Json::Value& request) {
        if (!request.isObject()) {
            return make_error_response("ADAPTER_ERROR", "request must be an object");
        }
        if (!request.isMember("timestamp_ns") || !request["timestamp_ns"].isInt64()) {
            return make_error_response("ADAPTER_ERROR", "missing timestamp_ns");
        }
        if (!request.isMember("image_jpeg_b64") || !request["image_jpeg_b64"].isString()) {
            return make_error_response("ADAPTER_ERROR", "missing image_jpeg_b64",
                                       request["timestamp_ns"].asInt64());
        }

        const int64_t timestamp_ns = request["timestamp_ns"].asInt64();
        std::vector<unsigned char> encoded_image;
        cv::Mat image_bgr;
        try {
            encoded_image = base64_decode(request["image_jpeg_b64"].asString());
            image_bgr = cv::imdecode(encoded_image, cv::IMREAD_COLOR);
        } catch (const std::exception& exc) {
            return make_error_response("ADAPTER_ERROR", std::string("image decode failed: ") + exc.what(),
                                       timestamp_ns);
        }
        if (image_bgr.empty()) {
            return make_error_response("ADAPTER_ERROR", "decoded image is empty", timestamp_ns);
        }

        std::vector<ORB_SLAM3::IMU::Point> imu_measurements;
        if (request.isMember("imu_samples") && request["imu_samples"].isArray()) {
            const Json::Value& imu_samples = request["imu_samples"];
            imu_measurements.reserve(imu_samples.size());
            for (const auto& imu_sample : imu_samples) {
                if (!imu_sample.isObject() || !imu_sample.isMember("timestamp_ns")) {
                    continue;
                }
                const int64_t imu_timestamp_ns = imu_sample["timestamp_ns"].asInt64();
                if (!imu_sample.isMember("angular_velocity_rad_s") ||
                    !imu_sample["angular_velocity_rad_s"].isArray() ||
                    imu_sample["angular_velocity_rad_s"].size() != 3) {
                    continue;
                }
                if (!imu_sample.isMember("linear_acceleration_m_s2") ||
                    !imu_sample["linear_acceleration_m_s2"].isArray() ||
                    imu_sample["linear_acceleration_m_s2"].size() != 3) {
                    continue;
                }
                cv::Point3f gyro(
                    imu_sample["angular_velocity_rad_s"][0].asFloat(),
                    imu_sample["angular_velocity_rad_s"][1].asFloat(),
                    imu_sample["angular_velocity_rad_s"][2].asFloat());
                cv::Point3f accel(
                    imu_sample["linear_acceleration_m_s2"][0].asFloat(),
                    imu_sample["linear_acceleration_m_s2"][1].asFloat(),
                    imu_sample["linear_acceleration_m_s2"][2].asFloat());
                const double imu_timestamp_s =
                    static_cast<double>(imu_timestamp_ns) / 1'000'000'000.0;
                imu_measurements.emplace_back(accel, gyro, imu_timestamp_s);
            }
        }
        std::sort(imu_measurements.begin(), imu_measurements.end(),
                  [](const ORB_SLAM3::IMU::Point& lhs, const ORB_SLAM3::IMU::Point& rhs) {
                      return lhs.t < rhs.t;
                  });

        Json::Value response(Json::objectValue);
        response["type"] = "orb_response";
        response["timestamp_ns"] = Json::Int64(timestamp_ns);
        response["camera_translation_xyz"] = Json::Value(Json::nullValue);
        response["camera_quaternion_wxyz"] = Json::Value(Json::nullValue);
        response["body_translation_xyz"] = Json::Value(Json::nullValue);
        response["body_quaternion_wxyz"] = Json::Value(Json::nullValue);
        response["tracking_image_jpeg_b64"] = Json::Value(Json::nullValue);
        response["error"] = Json::Value(Json::nullValue);

        if (last_frame_timestamp_ns_ >= 0 && timestamp_ns <= last_frame_timestamp_ns_) {
            response["tracking_state"] = "INVALID_FRAME_TIMESTAMP";
            response["error"] = "non-monotonic frame timestamp";
            return response;
        }
        if (imu_measurements.empty()) {
            response["tracking_state"] = "INSUFFICIENT_IMU";
            response["error"] = "empty imu_samples for frame";
            return response;
        }
        const double frame_timestamp_s = static_cast<double>(timestamp_ns) / 1'000'000'000.0;
        if (imu_measurements.back().t > frame_timestamp_s) {
            response["tracking_state"] = "INVALID_IMU_TIMESTAMP";
            response["error"] = "imu timestamp after frame timestamp";
            return response;
        }
        if (last_imu_timestamp_s_ > 0.0 && imu_measurements.front().t <= last_imu_timestamp_s_) {
            response["tracking_state"] = "INVALID_IMU_TIMESTAMP";
            response["error"] = "non-monotonic imu timestamps";
            return response;
        }

        try {
            std::lock_guard<std::mutex> lock(slam_mutex_);
            const Sophus::SE3f camera_pose =
                slam_.TrackMonocular(image_bgr, frame_timestamp_s, imu_measurements);

            const int tracking_state = slam_.GetTrackingState();
            response["tracking_state"] = tracking_state_to_string(tracking_state);
            last_frame_timestamp_ns_ = timestamp_ns;
            last_imu_timestamp_s_ = imu_measurements.back().t;

            if (is_finite_se3(camera_pose)) {
                fill_pose(&response, "camera_translation_xyz", "camera_quaternion_wxyz", camera_pose);
            }
        } catch (const std::exception& exc) {
            response["tracking_state"] = "ADAPTER_ERROR";
            response["error"] = std::string("slam exception: ") + exc.what();
        } catch (...) {
            response["tracking_state"] = "ADAPTER_ERROR";
            response["error"] = "unknown slam exception";
        }
        return response;
    }

   private:
    ORB_SLAM3::System slam_;
    std::mutex slam_mutex_;
    int64_t last_frame_timestamp_ns_ = -1;
    double last_imu_timestamp_s_ = -1.0;
};

struct Options {
    std::string host = "0.0.0.0";
    int port = 19090;
    std::string voc_file;
    std::string settings_file;
    bool use_viewer = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            options.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            options.port = std::stoi(argv[++i]);
        } else if (arg == "--voc-file" && i + 1 < argc) {
            options.voc_file = argv[++i];
        } else if (arg == "--settings-file" && i + 1 < argc) {
            options.settings_file = argv[++i];
        } else if (arg == "--use-viewer") {
            options.use_viewer = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: orb_native_adapter --voc-file <file> --settings-file <file> [--host 0.0.0.0]"
                   " [--port 19090] [--use-viewer]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.voc_file.empty()) {
        throw std::runtime_error("missing --voc-file");
    }
    if (options.settings_file.empty()) {
        throw std::runtime_error("missing --settings-file");
    }
    return options;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int create_listen_socket(const std::string& host, int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket() failed");
    }

    const int reuse = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        ::close(fd);
        throw std::runtime_error("setsockopt(SO_REUSEADDR) failed");
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        ::close(fd);
        throw std::runtime_error("invalid host: " + host);
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        throw std::runtime_error("bind() failed");
    }
    if (::listen(fd, 8) != 0) {
        ::close(fd);
        throw std::runtime_error("listen() failed");
    }
    return fd;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        const Options options = parse_options(argc, argv);
        if (!file_exists(options.voc_file)) {
            throw std::runtime_error("vocabulary file not found: " + options.voc_file);
        }
        if (!file_exists(options.settings_file)) {
            throw std::runtime_error("settings file not found: " + options.settings_file);
        }

        OrbNativeService service(options.voc_file, options.settings_file, options.use_viewer);
        const int server_fd = create_listen_socket(options.host, options.port);
        std::cout << "orb_native_adapter listening on " << options.host << ":" << options.port << std::endl;

        while (g_running) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            const int client_fd =
                ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            std::string request_json;
            if (recv_framed_message(client_fd, &request_json)) {
                Json::Value request;
                std::string parse_errors;
                Json::Value response;
                if (!parse_json(request_json, &request, &parse_errors)) {
                    response = make_error_response("ADAPTER_ERROR",
                                                   "invalid request JSON: " + parse_errors);
                } else {
                    response = service.process(request);
                }
                const std::string response_json = to_json_string(response);
                send_framed_message(client_fd, response_json);
            }
            ::close(client_fd);
        }

        ::close(server_fd);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "orb_native_adapter fatal error: " << exc.what() << std::endl;
        return 1;
    }
}
