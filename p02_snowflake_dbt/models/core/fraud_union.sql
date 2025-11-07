-- models/core/fraud_union.sql

select * from {{ ref('stg_fraud_data') }}
