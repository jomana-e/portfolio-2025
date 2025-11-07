with data as (
    select * from {{ ref('fraud_features') }}
)
select
    state,
    category,
    count(*) as total_transactions,
    sum(case when is_fraud = 1 then 1 else 0 end) as total_fraud,
    avg(amount) as avg_amount,
    sum(case when is_high_value = 1 then 1 else 0 end) as high_value_count
from data
group by 1,2
order by total_fraud desc
