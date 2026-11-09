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

## Author

Fjolla Qerimi
