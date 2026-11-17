# Databases

Introduction to relational (MySQL) and non-relational (MongoDB) databases,
part of the Holberton School Machine Learning curriculum (pipeline).

## Requirements

- Ubuntu 20.04 LTS
- MySQL 8.0 (8.0.39) for SQL files
- MongoDB 4.4 (4.4.29) for Mongo files
- Python 3.9 with PyMongo 4.6.2 for Python files

## Usage

```bash
cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
```

## Tasks

| # | File | Description |
|---|------|-------------|
| 0 | `0-create_database_if_missing.sql` | Creates the database `db_0` if it does not exist |
| 1 | `1-first_table.sql` | Creates the table `first_table` (`id`, `name`) |
| 2 | `2-list_values.sql` | Lists all rows of `first_table` |
| 3 | `3-insert_value.sql` | Inserts a new row in `first_table` |
| 4 | `4-best_score.sql` | Lists `score` and `name` of rows with `score >= 10`, best score first |
| 5 | `5-average.sql` | Computes the average score of `second_table` |
| 6 | `6-avg_temperatures.sql` | Average temperature by city, ordered by temperature (descending) |
| 7 | `7-max_state.sql` | Max temperature of each state, ordered by state name |
| 8 | `8-genre_id_by_show.sql` | Lists every show linked to at least one genre, with the genre id |
| 9 | `9-no_genre.sql` | Lists all shows without a genre |
| 10 | `10-count_shows_by_genre.sql` | Number of shows per genre |
| 11 | `11-rating_shows.sql` | Shows ranked by rating sum |
| 12 | `12-rating_genres.sql` | Genres ranked by rating sum |
| 13 | `13-uniq_users.sql` | Creates table `users` with a unique email |
| 14 | `14-country_users.sql` | Adds a `country` enumeration to `users` |
| 15 | `15-fans.sql` | Ranks band origins by number of fans |
| 16 | `16-glam_rock.sql` | Glam rock bands ranked by longevity |
| 17 | `17-store.sql` | Trigger decreasing item quantity after an order |
| 18 | `18-valid_email.sql` | Trigger resetting `valid_email` when the email changes |
| 19 | `19-bonus.sql` | Stored procedure `AddBonus` |
| 20 | `20-average_score.sql` | Stored procedure computing a user's average score |
| 21 | `21-div.sql` | Function `SafeDiv` |
| 22 | `22-list_databases` | Lists all MongoDB databases |
| 23 | `23-use_or_create_database` | Creates or uses the database `my_db` |
| 24 | `24-insert` | Inserts a document in `school` |
| 25 | `25-all` | Lists all documents in `school` |
| 26 | `26-match` | Lists documents with `name="Holberton school"` |
| 27 | `27-count` | Counts documents in `school` |
| 28 | `28-update` | Adds an `address` to matching documents |
| 29 | `29-delete` | Deletes matching documents |
| 30 | `30-all.py` | Lists all documents of a collection (PyMongo) |
| 31 | `31-insert_school.py` | Inserts a document from kwargs |
| 32 | `32-update_topics.py` | Updates the topics of a school |
| 33 | `33-schools_by_topic.py` | Lists schools having a given topic |
| 34 | `34-log_stats.py` | Stats about Nginx logs |
| 100 | `100-index_my_names.sql` | Index on the first letter of `name` |
| 101 | `101-index_name_score.sql` | Index on the first letter of `name` and `score` |
| 102 | `102-need_meeting.sql` | View `need_meeting` |
| 103 | `103-average_weighted_score.sql` | Procedure computing a weighted average score |
| 104 | `104-find` | Lists documents whose name starts with Holberton |
| 105 | `105-students.py` | Students sorted by average score |
| 106 | `106-log_stats.py` | Nginx log stats with top 10 IPs |

## Author

Fjolla Qerimi
