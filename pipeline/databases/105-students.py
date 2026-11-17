#!/usr/bin/env python3
"""Returns all students sorted by their average score"""


def top_students(mongo_collection):
    """Returns students with an averageScore key, sorted from best to worst"""
    project_step = {
        "$project": {
            "name": 1,
            "averageScore": {"$avg": "$topics.score"}
        }
    }
    sort_step = {
        "$sort": {"averageScore": -1}
    }
    pipeline = [project_step, sort_step]
    students = mongo_collection.aggregate(pipeline)
    return list(students)
