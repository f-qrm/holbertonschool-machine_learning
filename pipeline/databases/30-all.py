#!/usr/bin/env python3
"""Lists all documents in a MongoDB collection"""


def list_all(mongo_collection):
    """Returns a list of all documents in mongo_collection (empty if none)"""
    return list(mongo_collection.find())
