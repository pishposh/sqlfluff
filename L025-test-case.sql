-- https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#explicit_and_implicit_unnest

SELECT arr.r
FROM my_table AS m, UNNEST(m.arr) AS arr;

SELECT r
FROM my_table AS m, UNNEST(m.arr);

SELECT arr.r
FROM my_table AS m, m.arr AS arr;

SELECT r
FROM my_table AS m, m.arr;

{#

SELECT arr.r
FROM my_table AS m
CROSS JOIN UNNEST(m.arr) AS arr;

SELECT r
FROM my_table AS m
CROSS JOIN UNNEST(m.arr);

SELECT arr.r
FROM my_table AS m
CROSS JOIN m.arr AS arr;

SELECT r
FROM my_table AS m
CROSS JOIN m.arr; #}
