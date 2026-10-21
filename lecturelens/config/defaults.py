DEFAULT_LOCAL_MODEL: str = "gemma4:e4b"
DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text"
# The Redis consumer group used by the sync agent, and the prefix of its
# consumer names. The default keeps the historical name on purpose: renaming
# it would orphan the pending messages of every existing session, because the
# old group would never be read or acknowledged again. Set
# LECTURELENS_SYNC_CONSUMER_GROUP to move to a different name, and drain the
# old group first.
SYNC_CONSUMER_GROUP: str = "group:kuudere"
SYNC_CONSUMER_PREFIX: str = "consumer:kuudere"
