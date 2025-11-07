with source as (
    select *
    from PORTFOLIO_DB.RAW_DATA.RAW_FRAUD_DATA
),
renamed as (
    select
        cast(trans_date_trans_time as timestamp) as transaction_datetime,
        cc_num,
        merchant,
        category,
        amt as amount,
        first as first_name,
        last as last_name,
        gender,
        city,
        state,
        zip,
        city_pop,
        job,
        cast(dob as date) as date_of_birth,
        is_fraud
    from source
)
select * from renamed
