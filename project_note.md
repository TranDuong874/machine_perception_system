Upon receiving the packet server will do the following:

PerceptionPacket
|-> Get image -> OpenClip -> Embedding
|-> Extract Metadata: Yolo, Segment, Camera Pose and position -> Create metadata to store using embedding
|-> Extract SLAM track -> Run Depth -> Project point cloud -> Construct, Update, and Store occupancy map

LLM Service
|-> Receive user prompts, client should have prompt with timestamp to send to server instead of attaching it to perception packet -> Match time stamp with storage database -> Answer direct/present queries like: What am I looking at? Or where do I get to position X -> Query position match with frame (Edge case: How to handle query while person is moving?)
|-> Receive user prompts: Answering past question, probably do the same

Dont implement LLM Service and MCP tooling yet. We need to first, complete data transformation and storage pipeline on server first. 

Should we also use another database or something like that for query by attribute/metadata instead of query by the embedding vector?