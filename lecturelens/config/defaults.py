DEFAULT_LOCAL_MODEL: str = "gemma4:e4b"
DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text"
# The Redis consumer group used by the sync agent, and the prefix of its
# consumer names. They used to be hardcoded in three places.
SYNC_CONSUMER_GROUP: str = "group:kuudere"
SYNC_CONSUMER_PREFIX: str = "consumer:kuudere"
