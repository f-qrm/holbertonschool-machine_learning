#!/usr/bin/env python3
"""Inserts a new document in a MongoDB collection"""


def insert_school(mongo_collection, **kwargs):
    """Inserts a document built from kwargs and returns its new _id"""
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
