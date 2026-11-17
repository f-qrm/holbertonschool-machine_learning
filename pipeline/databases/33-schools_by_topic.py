#!/usr/bin/env python3
"""Finds schools having a specific topic in a MongoDB collection"""


def schools_by_topic(mongo_collection, topic):
    """Returns the list of schools that have topic in their topics"""
    return list(mongo_collection.find({"topics": topic}))
