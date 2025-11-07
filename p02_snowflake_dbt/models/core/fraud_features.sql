with base as (
    select * from {{ ref('fraud_union') }}
),
features as (
    select
        *,
        extract(hour from transaction_datetime) as transaction_hour,
        extract(dow from transaction_datetime) as transaction_day_of_week,
        case when amount > 200 then 1 else 0 end as is_high_value
    from base
)
select * from features
