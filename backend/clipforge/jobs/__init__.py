"""Job queue and worker pool.

Jobs live in SQLite so a crash never loses the queue; workers are threads inside
the app process by default and can also run as a separate ``clipforge worker``
process on a faster machine.
"""
