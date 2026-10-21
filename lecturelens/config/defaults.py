DEFAULT_LOCAL_MODEL: str = "gemma4:e4b"
DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text"
# The Redis consumer group used by the sync agent. It was hardcoded in three
# places, so a rename could silently split the group.
SYNC_CONSUMER_GROUP: str = "group:kuudere"
