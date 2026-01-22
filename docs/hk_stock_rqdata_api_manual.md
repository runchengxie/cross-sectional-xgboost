# 港股（公测版本）- RQData Python API

## 港交所股票合约基础信息

**API 传参 `market='hk'` 即可获取港交所合约数据**

### all_instruments - 获取所有合约基础信息

```python
rqdatac.all_instruments(type=None, market='hk', date=None)
```

获取港交所的所有股票合约信息。使用者可以通过这一方法很快地对合约信息有一个快速了解。可传入 `date` 筛选指定日期的合约，返回的 instrument 数据为合约的最新情况。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| type | *str* | 需要查询合约类型，例如：type='CS'代表股票。默认是所有类型 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 指定日期，筛选指定日期可交易的合约 |

**其中 type 参数传入的合约类型和对应的解释如下：**

| 合约类型 | 说明 |
| :--- | :--- |
| CS | Common Stock, 即股票 |

#### 返回

*pandas DataFrame* - 所有合约的基本信息。详细字段注释请参考 [instruments](#instruments---获取合约详细信息) 返回字段说明。

#### 范例

获取香港市场所有合约的基础信息：

```python
[In]rqdatac.all_instruments('CS',market='hk')
[Out]
     order_book_id      eng_symbol abbrev_symbol board_type    symbol listed_date de_listed_date  status  round_lot a_share_id exchange type trading_code      unique_id stock_connect
0       00001.XHKG    CKH HOLDINGS            CH  MainBoard        长和  1972-11-01     0000-00-00  Active      500.0       None     XHKG   CS        00001  00001_01.XHKG     sh_and_sz
1       00002.XHKG    CLP HOLDINGS          ZDKG  MainBoard      中电控股  1980-01-02     0000-00-00  Active      500.0       None     XHKG   CS        00002  00002_01.XHKG     sh_and_sz
2       00003.XHKG  HK & CHINA GAS        XGZHMQ  MainBoard    香港中华煤气  1960-04-11     0000-00-00  Active     1000.0       None     XHKG   CS        00003  00003_01.XHKG     sh_and_sz
3       00004.XHKG  WHARF HOLDINGS         JLCJT  MainBoard     九龙仓集团  1921-01-01     0000-00-00  Active     1000.0       None     XHKG   CS        00004  00004_01.XHKG     sh_and_sz
4       00005.XHKG   HSBC HOLDINGS          HFKG  MainBoard      汇丰控股  1980-01-02     0000-00-00  Active      400.0       None     XHKG   CS        00005  00005_01.XHKG     sh_and_sz
...            ...             ...           ...        ...       ...         ...            ...     ...        ...        ...      ...  ...          ...            ...           ...
3309    83690.XHKG      MEITUAN-WR          MTWR  MainBoard     美团-WR  2023-06-19     0000-00-00  Active      100.0       None     XHKG   CS        83690  83690_01.XHKG
3310    86618.XHKG     JD HEALTH-R         JDJKR  MainBoard    京东健康-R  2023-06-19     0000-00-00  Active       50.0       None     XHKG   CS        86618  86618_01.XHKG
3311    89618.XHKG          JD-SWR       JDJTSWR  MainBoard  京东集团-SWR  2023-06-19     0000-00-00  Active       50.0       None     XHKG   CS        89618  89618_01.XHKG
3312    89888.XHKG        BIDU-SWR       BDJTSWR  MainBoard  百度集团-SWR  2023-06-19     0000-00-00  Active       50.0       None     XHKG   CS        89888  89888_01.XHKG
3313    89988.XHKG         BABA-WR        ALBBWR  MainBoard   阿里巴巴-WR  2023-06-19     0000-00-00  Active      100.0       None     XHKG   CS        89988  89988_01.XHKG

[3314 rows x 15 columns]
```

---

### instruments - 获取合约详细信息

```python
rqdatac.instruments(order_book_ids, market='hk')
```

获取港交所某一个或多个股票最新的详细信息。

> **注意事项**
>
> 目前系统并不支持跨市场的同时调用。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* OR *str list* | 合约代码，可传入 order_book_id, order_book_id list。<br>港交所股票的 order_book_id 通常类似'00001.XHKG'。 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

一个 instrument 对象，或一个 instrument list。

**股票 Instrument 对象**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_id | *str* | 证券代码，证券的独特的标识符。港股以'.XHKG'结尾。 |
| symbol | *str* | 证券的简称，例如'长和' |
| abbrev_symbol | *str* | 证券的名称缩写，在中国 A 股就是股票的拼音缩写。例如：'CH'就是长和股票的证券拼音名缩写 |
| eng_symbol | *str* | 证券的英文名称。例如：'CKH HOLDINGS'就是长和股票的英文名称 |
| round_lot | *int* | 一手对应多少股 |
| listed_date | *str* | 该证券上市日期 |
| de_listed_date | *str* | 退市日期 |
| type | *str* | 合约类型，目前支持的类型有: 股票:'CS' |
| exchange | *str* | 交易所，'XHKG' - 港交所 |
| board_type | *str* | 板块类别，'MainBoard' - 主板,'GEM' - 创业板 |
| status | *str* | 合约状态。'Active' - 正常上市, 'Delisted' - 终止上市 |
| stock_connect | *str* | 沪深港通标识。<br>'sh_and_sz':沪深港通<br>'sz':'深港通'<br>'sh':沪港通 |
| a_share_id | *str* | 对应 A 股 order_book_id |
| trading_code | *str* | 交易代码 |
| unique_id | *str* | 米筐内部编码，因为港股存在代码复用，所以米筐内部用这个编码作为合约的唯一标识，如_02 代表复用一次（一般用户可以不关注这个字段） |

#### 范例

获取单一股票合约的详细信息：

```python
In [5]: rqdatac.instruments('00013.XHKG',market='hk')
Out[5]:
Instrument(order_book_id='00013.XHKG', eng_symbol='HUTCHMED', abbrev_symbol='HHYY', board_type='MainBoard', symbol='和黄医药', listed_date='2021-06-30', de_listed_date='0000-00-00', status='Active', round_lot=500.0, exchange='XHKG', type='CS', trading_code='00013', unique_id='00013_02.XHKG', stock_connect='sh_and_sz')
```

---

### get_ex_factor - 获取复权因子

```python
get_ex_factor(order_book_ids, start_date=None, end_date=None, market='hk')
```

获取某只股票或股票列表在一段时间内的复权因子（包含起止日期，以除权除息日为查询基准）。如未指定日期，则默认所有。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | 合约代码 |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame* - 包含了复权因子的日期和对应的各项数值

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| ex_date | *pandas.Timestamp* | 除权除息日 |
| ex_factor | *float* | 复权因子，考虑了分红派息与拆分的影响，为一段时间内的股价调整乘数。<br>举例来说，长和（'00001.XHKG'）在 2024 年 9 月 13 日每股派发现金股利港币 0.688 元。<br>9 月 12 日的收盘价为 41.85 元，其除权除息后的价格应当为 (41.85-0.688) / 1 = 41.162.本期复权因子为 41.85 / 41.162 = 1.016714 |
| ex_cum_factor | *float* | 累计复权因子，X 日所在期复权因子 = 当前最新累计复权因子 / 截至 X 日最新累计复权因子。<br>长和（'00001.XHKG'）2024 年 9 月 13 日所在期复权因子 = 4.117329 / 4.049641 = 1.016714 |
| announcement_date | *pandas.Timestamp* | 股权登记日 |
| ex_end_date | *pandas.Timestamp* | 复权因子所在期的截止日期 |

#### 范例

```python
[In]
rqdatac.get_ex_factor('00001.XHKG',market='hk')
[Out]
            ex_cum_factor ex_end_date  ex_factor announcement_date order_book_id
ex_date
1999-10-08       1.005224  2000-05-16   1.005224        1999-10-07    00001.XHKG
2000-05-16       1.018303  2000-10-10   1.013011        2000-05-15    00001.XHKG
2000-10-10       1.022481  2001-05-15   1.004103        2000-10-09    00001.XHKG
2001-05-15       1.036414  2001-10-09   1.013627        2001-05-14    00001.XHKG
...              ...       ...          ...             ...           ...
2024-05-28       4.049641  2024-09-13   1.044739        2024-05-27    00001.XHKG
2024-09-13       4.117329         NaT   1.016714               NaT    00001.XHKG
```

---

### get_exchange_rate - 获取汇率信息

```python
get_exchange_rate(start_date=None, end_date=None, fields=None)
```

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期 |
| fields | *list* | 字段名称，如需计算财务数据，可指定 currency_pair 和 middle_referrence_rate |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| date | *pandas.Timestamp* | 时间戳 |
| currency_pair | *str* | 货币对。返回值见下方，如'HKDCNY'表示 1 港币对应的人民币 |
| middle_referrence_rate | *str* | 中间价，香港金融管理局披露值，**月更新** |
| bid_referrence_rate | *str* | 买入参考汇率，上交所和深交所披露值，日更新 (仅限 HKDCNY) |
| ask_referrence_rate | *str* | 卖出参考汇率，上交所和深交所披露值，日更新 (仅限 HKDCNY) |
| bid_settlement_rate_sh | *str* | 买入结算汇率-沪港通，盘后更新 (仅限 HKDCNY) |
| ask_settlement_rate_sh | *str* | 卖出结算汇率-沪港通，盘后更新 (仅限 HKDCNY) |
| bid_settlement_rate_sz | *str* | 买入结算汇率-深港通，盘后更新 (仅限 HKDCNY) |
| ask_settlement_rate_sz | *str* | 卖出结算汇率-深港通，盘后更新 (仅限 HKDCNY) |

**currency_pair 返回值：**

| 货币单位 | 货币对 |
| :--- | :--- |
| 美元 | HKDUSD |
| 日本元 | HKDJPY |
| 澳门元 | HKDMOP |
| 新加坡元 | HKDSGD |
| 泰国铢 | HKDTHB |
| 人民币元 | HKDCNY |
| 台湾元 | HKDTWD |
| 欧元 | HKDEUR |
| 加拿大元 | HKDCAD |
| 澳大利亚元 | HKDAUD |
| 马来西亚林吉特 | HKDMYR |
| 英镑 | HKDGBP |
| 南非兰特 | HKDZAR |
| 印度尼西亚卢比 | HKDIDR |

#### 范例

*   获取 HKDCNY 20250101 - 20250630 所有字段

```python
[In]
df = get_exchange_rate(20250101,20250630)
df[df['currency_pair'] =='HKDCNY']
[Out]
            currency_pair bid_referrence_rate ask_referrence_rate ... ask_settlement_
date
2025-01-02        HKDCNY              0.9165              0.9731 ...          0.94448
2025-01-03        HKDCNY              0.9137              0.9703 ...          0.94258
2025-01-04        HKDCNY                 NaN                 NaN ...              NaN
2025-01-06        HKDCNY              0.9175              0.9743 ...          0.94597
2025-01-07        HKDCNY              0.9174              0.9742 ...          0.94545
...
2025-06-25        HKDCNY              0.8868              0.9416 ...          0.91418
2025-06-26        HKDCNY              0.8862              0.9410 ...          0.91358
2025-06-27        HKDCNY              0.8852              0.9400 ...          0.91261
2025-06-28        HKDCNY                 NaN                 NaN ...              NaN
2025-06-30        HKDCNY              0.8858              0.9406 ...          0.91319
```

*   获取日期为 20250210 1 港币对应的所有货币汇率

```python
[In]
get_exchange_rate(20250210,20250210,fields =['currency_pair','middle_referrence_rate'])
[Out]
           currency_pair  middle_referrence_rate
date
2025-02-10        HKDAUD                  0.2046
2025-02-10        HKDCAD                  0.1841
2025-02-10        HKDCNY                  0.9384
2025-02-10        HKDEUR                  0.1244
2025-02-10        HKDJPY                 19.5065
2025-02-10        HKDMYR                  0.5738
2025-02-10        HKDSGD                  0.1738
2025-02-10        HKDTHB                  4.3403
2025-02-10        HKDTWD                  4.1152
2025-02-10        HKDUSD                  0.1284
2025-02-10        HKDGBP                  0.1035
```

---

### get_shares - 获取流通股信息

```python
get_shares(order_book_ids, start_date=None, end_date=None, fields=None, market='hk', expect_df=True)
```

获取股票或股票列表在一段时间内的流通情况（包含起止日期）。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | 可输入 order_book_id 或 symbol |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期， |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期，不传入 start_date ,end_date 则 默认返回最近三个月的数据 |
| fields | *str* OR *str list* | 默认为所有字段。见下方列表 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |
| expect_df | *boolean* | 默认返回 pandas dataframe,如果调为 False ,则返回原有的数据结构 |

#### 返回

*pandas  DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| total | *float* | 总股本 |
| authorized_shares | *float* | 法定股数(股) |
| total_a | *float* | A 股总股本 |
| not_hk_shares | *float* | 非港股股数(股) |
| preferred_shares | *float* | 优先股 |
| total_hk | *float* | 已上市港股股数(股) |
| total_hk1 | *float* | 可在港股交易的股数(股) |

#### 范例

获取平安银行流通股概况

```python
[In]
get_shares(order_book_ids='00038.XHKG', start_date=20250715, end_date=20250720, market='hk')
[Out]
                     authorized_shares  not_hk_shares  preferred_shares         total       total_a
order_book_id date
00038.XHKG 2025-07-15     1.123645e+09    731705275.0               0.0  1.123645e+09   731705275.0
           2025-07-16     1.123645e+09    731705275.0               0.0  1.123645e+09   731705275.0
           2025-07-17     1.123645e+09    731705275.0               0.0  1.123645e+09   731705275.0
           2025-07-18     1.123645e+09    731705275.0               0.0  1.123645e+09   731705275.0
```

---

### get_industry - 获取某行业的股票列表

```python
get_industry(industry, source='citics_2019', date=None, market='hk')
```

通过传入行业名称、行业指数代码或者行业代号，拿到指定行业的股票列表

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| industry | *str* | **必填参数**，可传入行业名称、行业指数代码或者行业代号 |
| source | *str* | 分类依据。 citics_2019:中信 2019 分类，sws_2021:申万行业分类，hsi:恒生行业分类。 默认 source='citics_2019' |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认为当前最新日期 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*list*

#### 范例

得到当前某一级行业的股票列表：

```python
[In]
get_industry('原材料业',source='hsi',market='hk')
[Out]
['00094.XHKG',
 '00098.XHKG',
 '00159.XHKG',
 '00166.XHKG',
 '00189.XHKG',
 '00195.XHKG',
 '00217.XHKG',
 '00235.XHKG',
 '00274.XHKG',
 '00297.XHKG',
 '00301.XHKG',
 '00323.XHKG',
 '00338.XHKG',
 '00340.XHKG',
 '00347.XHKG',
 '00358.XHKG',
 '00362.XHKG',
 '00372.XHKG',
...
]
```

---

### get_industry_change - 获取某行业的股票纳入剔除日期

```python
get_industry_change(industry, source='citics_2019', level=None, market='hk')
```

通过传入行业名称、行业指数代码或者行业代号，拿到指定行业的股票纳入剔除日期

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| industry | *str* | **必填参数**，可传入行业名称、行业指数代码或者行业代号 |
| source | *str* | 分类依据。 citics_2019:中信 2019 分类，sws_2021:申万行业分类，hsi:恒生行业分类。默认 source='citics_2019' |
| level | *integer* | 行业分类级别，共三级，默认一级分类。参数 1,2,3 一一对应 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| start_date | *pandas.Timestamp* | 起始日期 |
| cancel_date | *pandas.Timestamp* | 取消日期，2200-12-31 表示未披露 |

#### 范例

得到当前某一级行业的股票纳入剔除日期：

```python
[In]
get_industry_change(industry='原材料业', level=1,source='hsi',market='hk')
[Out]
               start_date cancel_date
order_book_id
01812.XHKG     2013-09-09  2200-12-31
00347.XHKG     2013-09-09  2200-12-31
00358.XHKG     2013-09-09  2200-12-31
01787.XHKG     2018-09-28  2200-12-31
00338.XHKG     2013-09-09  2200-12-31
...            ...         ...
09879.XHKG     2024-03-21  2200-12-31
06616.XHKG     2021-07-16  2200-12-31
02237.XHKG     2022-07-18  2200-12-31
02881.XHKG     2024-06-18  2200-12-31
02610.XHKG     2025-03-25  2200-12-31
```

---

### get_instrument_industry - 获取股票的指定行业分类

```python
get_instrument_industry(order_book_ids, source='citics_2019', level=1, date=None, market='hk')
```

通过 order_book_id 传入，拿到某个日期的该股票指定的行业分类

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_id | *str* or *str list* | **必填参数**，股票合约代码，可输入 order_book_id, order_book_id list |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认为当前最新日期 |
| source | *str* | 分类依据。citics_2019:中信 2019 分类，sws_2021:申万行业分类，hsi:恒生行业分类。 默认 source='citics_2019'. |
| level | *integer* | 行业分类级别，共三级，默认返回一级分类。参数 0,1,2,3 一一对应，其中 0 返回三级分类完整情况 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| first_industry_code | *str* | 一级行业代码 |
| first_industry_name | *str* | 一级行业名称 |
| second_industry_code | *str* | 二级行业代码 |
| second_industry_name | *str* | 二级行业名称 |
| third_industry_code | *str* | 三级行业代码 |
| third_industry_name | *str* | 三级行业名称 |

#### 范例

得到当前股票所对应的一级行业：

```python
[In]
get_instrument_industry(order_book_ids='00001.XHKG',market='hk')
[Out]
              first_industry_code first_industry_name
order_book_id
00001.XHKG                     43            综合金融
```

得到当前股票组所对应的中信行业的全部分类：

```python
[In]
get_instrument_industry(['00001.XHKG','00038.XHKG'],source='citics_2019',level=0,market='hk')
[Out]
              first_industry_code first_industry_name second_industry_code second_industry_name ...
order_book_id
00038.XHKG                     26                机械               2620             专用机械 ...
00001.XHKG                     43            综合金融               4320         多领域控股Ⅱ ...
```

---

### get_industry_mapping - 获取行业分类概览

```python
get_industry_mapping(source='citics_2019', date=None, market='hk')
```

通过传入分类依据，获得对应的一二三级行业代码和名称。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| source | *str* | 分类依据。 citics_2019:中信 2019 分类，sws_2021:申万行业分类，hsi:恒生行业分类。默认 source='citics_2019' |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认为当前最新日期 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| first_industry_code | *str* | 一级行业代码 |
| first_industry_name | *str* | 一级行业名称 |
| second_industry_code | *str* | 二级行业代码 |
| second_industry_name | *str* | 二级行业名称 |
| third_industry_code | *str* | 三级行业代码 |
| third_industry_name | *str* | 三级行业名称 |

#### 范例

得到当前行业分类的概览：

```python
[In]
get_industry_mapping(market='hk')
[Out]
  first_industry_code first_industry_name second_industry_code second_industry_name
0                  10            石油石化                 1010             石油开采Ⅱ
1                  10            石油石化                 1020             石油化工
2                  10            石油石化                 1020             石油化工
3                  10            石油石化                 1030             油服工程
4                  10            石油石化                 1030             油服工程
...
```

---

### get_turnover_rate - 获取历史换手率

```python
get_turnover_rate(order_book_ids, start_date=None, end_date=None, fields=None, expect_df=True, market='hk')
```

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | **必填参数**，合约代码，可输入 order_book_id, order_book_id list |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期，不传入 start_date ,end_date 则 默认返回最近三个月的数据 |
| fields | *str* OR *str list* | 默认为所有字段。当天换手率 - `today`，过去一周平均换手率 - `week`，过去一个月平均换手率 - `month`，过去一年平均换手率 - `year`，当年平均换手率 - `current_year` |
| expect_df | *boolean* | 默认返回 pandas dataframe。如果调为 False，则返回原有的数据结构 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

#### 范例

获取长和 00001.XHKG 历史换手率情况：

```python
[In]
get_turnover_rate('00001.XHKG',20250801,20250810,market='hk')
[Out]
                                 today    week   month    year  current_year
order_book_id  tradedate
00001.XHKG     2025-08-01       0.2842  0.2667  0.2035  0.2562        0.3156
               2025-08-04       0.2290  0.2372  0.2088  0.2566        0.3150
               2025-08-05       0.1249  0.2097  0.2050  0.2564        0.3137
               2025-08-06       0.1689  0.2133  0.2034  0.2564        0.3127
               2025-08-07       0.1340  0.1882  0.2046  0.2564        0.3115
               2025-08-08       0.2370  0.1788  0.2083  0.2570        0.3110
```

获取多支股票一段时间内的周平均换手率

```python
[In]
get_turnover_rate(['00001.XHKG', '00038.XHKG'], '20250804', '20250805', 'week', market='hk')
[Out]
                                  week
order_book_id  tradedate
00001.XHKG     2025-08-04       0.2372
               2025-08-05       0.2097
00038.XHKG     2025-08-04       1.1502
               2025-08-05       1.1519
```

---

### hk.get_southbound_eligible_secs - 获取港股通成分股数据

```python
hk.get_southbound_eligible_secs(trading_type='sh', date=None, start_date=None, end_date=None, market='hk')
```

> **注意事项**
>
> 请先单独安装 rqdatac_hk，导入后使用

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| trading_type | *str* | **必填参数**，支持填入 'sh':'港股通（沪）'sz':'港股通（深）' |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认为最新记录日期 |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 指定开始日期，不能和 date 同时指定 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 指定结束日期, 需和 start_date 同时指定并且应当不小于开始日期 |
| market | *str* | 市场，仅限'hk'香港市场 |

#### 返回

*某一天港股通成分股的 order_book_id list*

#### 范例

*   指定 date 获取某一天的 sh 港股通成分股数据

```python
[In]
import rqdatac_hk
import rqdatac
rqdatac.init()
rqdatac.hk.get_southbound_eligible_secs(trading_type='sh',date=20250929)
[Out]
['00001.XHKG',
 '00002.XHKG',
 '00003.XHKG',
 .....
]
```

*   获取某段时间的 sz 港股通成分股数据

```python
[In]
rqdatac.hk.get_southbound_eligible_secs(trading_type='sz',start_date=20250925,end_date=20250929)
[Out]
[{datetime.datetime(2025, 9, 25, 0, 0): ['00001.XHKG',
 '00002.XHKG',
 '00003.XHKG',
 '00004.XHKG'
 ...
 ],
 datetime.datetime(2025, 9, 29, 0, 0): ['00001.XHKG',
 '00002.XHKG',
 '00003.XHKG',
 '00004.XHKG',
 ...]}
]
```

---

## 港交所股票行情（基于港交所延时行情）

### get_price - 获取合约历史行情数据

```python
get_price(order_book_ids, start_date=None, end_date=None, frequency='1d', fields=None, adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn')
```

获取指定港股合约或合约列表的历史数据（包含起止日期，周线、日线、分钟线或 tick）。

> **注意事项**
>
> 周线数据目前只支持'1w',依据日线数据进行合成，例如股票周线的前复权数据使用前复权日线数据进行合成，股票周线的不复权数据使用不复权的日线数据合成。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* OR *str list* | **必填参数**，合约代码，可传入 order_book_id, order_book_id list |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期 |
| frequency | *str* | 历史数据的频率。 现在支持**周/日/分钟/tick 级别**的历史数据，默认为'1d'。<br> 1m - 分钟线<br> 1d - 日线 <br>1w - 周线，只支持'1w' <br> 日线和分钟可选取不同频率，例如'5m'代表 5 分钟线。 |
| fields | *str* OR *str list* | 字段名称 |
| adjust_type | *str* | 权息修复方案，默认为`pre`。<br>不复权 - `none`，<br>前复权 - `pre`，后复权 - `post`，<br>前复权 - `pre_volume`, 后复权 - `post_volume` <br>两组前后复权方式仅 volume 字段处理不同，其他字段相同。其中'pre'、'post'中的 volume 采用拆分因子调整；'pre_volume'、'post_volume'中的 volume 采用复权因子调整。 |
| skip_suspended | *bool* | 是否跳过停牌数据。默认为 False，不跳过，用停牌前数据进行补齐。True 则为跳过停牌期。 |
| expect_df | *bool* | 默认返回 pandas dataframe。如果调为 False，则返回原有的数据结构,周线数据需设置 expect_df=True |
| time_slice | *str, datetime.time* | 开始、结束时间段。默认返回当天所有数据。<br>支持分钟 / tick 级别的切分，详见下方范例。 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

**bar 数据**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| open | *float* | 开盘价 |
| close | *float* | 收盘价 |
| high | *float* | 最高价 |
| low | *float* | 最低价 |
| limit_up | *float* | 涨停价，港股该字段为 nan |
| limit_down | *float* | 跌停价，港股该字段为 nan |
| total_turnover | *float* | 成交额 |
| volume | *float* | 成交量 |
| num_trades | *int* | 成交笔数 ，港股该字段为 nan |
| prev_close | *float* | 昨日收盘价 （交易所披露的原始昨收价，复权方法对该字段无效） |

**tick 数据**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| datetime | *pandas.Timestamp* | 交易所时间戳 |
| open | *float* | 当日开盘价 |
| high | *float* | 当日最高价 |
| low | *float* | 当日最低价 |
| last | *float* | 最新价 |
| prev_close | *float* | 昨日收盘价 |
| total_turnover | *float* | 当天累计成交额 |
| volume | *float* | 当天累计成交量 |
| num_trades | *int* | 成交笔数，港股该字段为 nan |
| limit_up | *float* | 涨停价，港股该字段为 nan |
| limit_down | *float* | 跌停价，港股该字段为 nan |
| a1~a10 | *float* | 卖一至十档报盘价格 |
| a1_v~a10_v | *float* | 卖一至十档报盘量 |
| b1~b10 | *float* | 买一至十档报盘价 |
| b1_v~b10_v | *float* | 买一至十档报盘量 |
| change_rate | *float* | 涨跌幅 |
| trading_date | *pandas.Timestamp* | 交易日期 |

#### 范例

获取单一股票 20250101 - 20250301 的前复权日行情（返回pandas DataFrame）:

```python
[In] rqdatac.get_price('00013.XHKG',20250201,20250301,'1d',adjust_type='pre',market='hk')
[Out]
                      limit_up   high      volume  close  prev_close  limit_down
order_book_id date
00013.XHKG    2025-02-03       NaN  21.00   1991902.0  20.50       20.90         NaN
              2025-02-04       NaN  21.10   1776500.0  20.95       20.50         NaN
              2025-02-05       NaN  21.95   3349788.0  21.45       20.95         NaN
              2025-02-06       NaN  22.25   3913626.0  22.15       21.45         NaN
              2025-02-07       NaN  22.15  10548253.0  20.95       22.15         NaN
              2025-02-10       NaN  21.20   6369895.0  21.00       20.95         NaN
              2025-02-11       NaN  21.20   3942204.0  20.70       21.00         NaN
              2025-02-12       NaN  20.85   7415758.0  20.25       20.70         NaN
              2025-02-13       NaN  20.80   6638000.0  20.35       20.25         NaN
              2025-02-14       NaN  21.30   8699765.0  21.30       20.35         NaN
              2025-02-17       NaN  22.05   7436135.0  21.25       21.30         NaN
              2025-02-18       NaN  21.50   6783079.0  21.50       21.25         NaN
              2025-02-19       NaN  23.10  10917076.0  22.95       21.50         NaN
              2025-02-20       NaN  24.60  17384900.0  23.80       22.95         NaN
              2025-02-21       NaN  25.70  14068700.0  25.45       23.80         NaN
              2025-02-24       NaN  25.70  12209181.0  24.80       25.45         NaN
              2025-02-25       NaN  26.20   9420500.0  24.80       24.80         NaN
              2025-02-26       NaN  26.50  10825094.0  26.10       24.80         NaN
              2025-02-27       NaN  27.15  10507606.0  26.70       26.10         NaN
              2025-02-28       NaN  26.60   9462325.0  25.75       26.70         NaN
```

获取单一股票 20250101 - 20250301 的后复权分钟行情（返回pandas DataFrame）:

```python
[In] rqdatac.get_price('00001.XHKG',20250101,20250301,'1m',adjust_type='post',market='hk')
[Out]
                                  high    volume     close  num_trades       low
order_book_id datetime
00001.XHKG    2025-01-02 09:31:00 170.8691  126234.0  169.6339         NaN  169.22
              2025-01-02 09:32:00 169.6339  321500.0  169.4281         NaN  168.39
              2025-01-02 09:33:00 169.4281       0.0  169.4281         NaN  169.42
              2025-01-02 09:34:00 169.4281    6500.0  169.4281         NaN  169.22
              2025-01-02 09:35:00 169.2222  116500.0  169.2222         NaN  168.39
...                                    ...       ...       ...         ...     ...
              2025-02-28 15:56:00 160.5758   58000.0  160.5758         NaN  160.37
              2025-02-28 15:57:00 160.7817  263000.0  160.5758         NaN  160.57
              2025-02-28 15:58:00 160.7817   46999.0  160.5758         NaN  159.34
              2025-02-28 15:59:00 160.5758   20803.0  160.3700         NaN  160.37
              2025-02-28 16:00:00 160.5758 8032500.0  159.9582         NaN  159.95

[12690 rows x 7 columns]
```

获取单一股票 20250303 的 tick 行情（返回pandas DataFrame）:

```python
[In] rqdatac.get_price('00001.XHKG',20250303,20250303,'tick',market='hk')
[Out]
                                     trading_date   open   last   high    low  prev_close
order_book_id datetime
00001.XHKG    2025-03-03 09:20:38.791  2025-03-03  39.05  39.05  39.05  39.05       39.10
              2025-03-03 09:30:00.107  2025-03-03  39.05  39.05  39.05  39.05       39.10
              2025-03-03 09:30:00.150  2025-03-03  39.05  39.10  39.10  39.05       39.10
              2025-03-03 09:30:00.151  2025-03-03  39.05  39.15  39.15  39.05       39.10
              2025-03-03 09:30:01.026  2025-03-03  39.05  39.15  39.15  39.05       39.10
...                                           ...    ...    ...    ...    ...         ...
              2025-03-03 15:59:38.721  2025-03-03  39.05  39.15  39.55  38.90       39.10
              2025-03-03 15:59:49.550  2025-03-03  39.05  39.15  39.55  38.90       39.10
              2025-03-03 15:59:50.546  2025-03-03  39.05  39.10  39.55  38.90       39.10
              2025-03-03 15:59:51.898  2025-03-03  39.05  39.15  39.55  38.90       39.10
              2025-03-03 16:08:13.506  2025-03-03  39.05  39.15  39.55  38.90       39.10

[1317 rows x 52 columns]
```

---

## 港股财务数据

### get_pit_financials_ex - 查询季度财务信息(point-in-time 形式)

```python
get_pit_financials_ex(order_book_ids, fields, start_quarter, end_quarter, date=None, statements='latest', market='cn')
```

以给定一个报告期回溯的方式获取季度基础财务数据（三大表），即利润表，资产负债表，现金流量表。

> **注意事项**
>
> 该 API 返回的因子值均为做了港币汇率转换后的值，除了货币单位为港币的合约，其余并非财报披露的原始值。若需获取原始值，请参考 [hk.get_detailed_financial_items-params](#rqdata-API-hk_get_detailed_financial_items-params)

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | **必填参数**，合约代码，可传入 order_book_id, order_book_id list ，该参数必填 |
| fields | *list* | **必填参数**，需要传入的财务字段。支持的字段仅限**利润表、资产负债表、现金流量表三大表字段**，具体字段见下方返回。 |
| start_quarter | *str* | **必填参数**，财报回溯查询的起始报告期，例如'2015q2'代表 2015 年半年报， 该参数必填 。 |
| end_quarter | *str* | **必填参数**，财报回溯查询的截止报告期，例如'2015q4'代表 2015 年年报，该参数必填。 |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认查询日期为当前最新日期 |
| statements | *str* | 基于查询日期，返回某一个报告期的所有记录或最新一条记录，设置 statements 为 all 时返回所有记录，statements 等于 latest 时返回最新的一条记录，默认为 latest. |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| quarter | *str* | 报告期 |
| info_date | *pandas.Timestamp* | 公告发布日 |
| fields | *list* | 返回的财务字段。返回的字段仅限**利润表、资产负债表、现金流量表三大表字段**，具体字段见下方返回。 |
| if_adjusted | *int* | 是否为非当期财报数据, 0 代表当期，1 代表非当期（比如 18 年的财报会披露本期和上年同期的数值，17 年年报的财务数值在 18 年年报中披露的记录则为非当期， 17 年年报的财务数值在 17 年年报中披露则为当期）。 |
| fiscal_year | *pandas.Timestamp* | 财政年度 |
| standard | *str* | 会计准则，中国会计准则、非中国会计准则_金融公司、非中国会计准则_保险公司、非中国会计准则_非金融非保险公司 |

#### 财务报表字段列表

##### 中国会计准则

**利润表**

| 字段 | 释义、备注 |
| :--- | :--- |
| other_income_equity_classified_income_statement | 2.1 权益法下在被投资单位以后将重分类进损益的其他综合 |
| remearsured_other_income | 1.1 重新计量设定收益计划净负债或净资产的变动 |
| other_income_equity_unclassified_income_statement | 1.2 权益法下在被投资单位不能重分类进损益的其他综合 |
| other_debt_investment_change | 2.7 其他债权投资公允价值变动 |
| assets_reclassified_other_income | 2.8 金融资产重分类计入其他综合收益的金额 |
| other_equity_instruments_change | 1.3 其他权益工具投资公允价值变动 |
| other_debt_investment_reserve | 2.9 其他债权投资信用减值准备 |
| corporate_credit_risk_change | 1.4 企业自身信用风险公允价值变动 |
| cash_flow_hedging_effective_portion | 2.4 现金流量套期损益的有效部分 |
| foreign_currency_statement_converted_difference | 2.5 外币财务报表折算差额 |
| others | 2.6 其他（以后能重分类进损益表的其他综合收益） |

**资产负债表**

| 字段 | 释义、备注 |
| :--- | :--- |
| use_right_assets | 使用权资产 |
| unearned_reserve_receivable | 应收分保未到期责任准备金 |
| undistributed_profit | 未分配利润 |
| unclaimed_reserve_receivable | 应收分保未决赔款准备金 |
| unclaimed_indemnity_reserve | 未决赔款准备金 |
| uncertained_premium_reserve | 未到期责任准备金 |
| treasury_stock | 减：库存股 |
| total_liabilities | 负债合计 |
| total_fixed_assets | 固定资产 |
| total_equity_and_liabilities | 负债和所有者（或股东权益）总计 |
| total_equity | 所有者权益（或股东权益）合计 |
| total_assets | 资产总计 |
| tax_payable | 应交税费 |
| surplus_reserve | 盈余公积 |
| sub_issue_security_proceeds | 代理承销证券款 |
| specific_reserve | 专项储备 |
| short_term_loans | 短期借款 |

**现金流量表**

| 字段 | 释义、备注 |
| :--- | :--- |
| net_increase_from_other_financial_institutions | 向其他金融机构拆入资金净增加额 |
| net_increase_from_central_bank | 向中央银行借款净增加额 |
| cash_paid_for_policy_dividends | 支付保单红利的现金 |
| cash_paid_for_taxes | 支付的各项税费 |
| cash_paid_for_employee | 支付给职工以及为职工支付的现金 |
| cash_paid_for_other_financing_activities | 支付其他与筹资活动有关的现金 |
| cash_paid_for_other_operation_activities | 支付其他与经营活动有关的现金 |
| cash_paid_for_other_investment_activities | 支付其他与投资活动有关的现金 |
| cash_paid_for_orignal_insurance | 支付原保险合同赔付款项的现金 |
| net_increase_in_pledge_loans | 质押贷款净增加额 |
| cash_paid_for_asset | 购建固定资产、无形资产和其他长期资产支付的现金 |
| cash_paid_for_goods_and_services | 购买商品、接受劳务支付的现金 |
| cash_received_from_sales_of_goods | 销售商品、提供劳务收到的现金 |
| cash_received_from_investors | 吸收投资收到的现金 |
| cash_equivalent_increase | 现金及现金等价物净增加额 |
| cash_paid_to_acquire_investment | 投资支付的现金 |

##### 非中国会计准则_金融公司

**利润表**

| 字段 | 释义、备注 |
| :--- | :--- |
| net_interest_income | 净利息收入 |
| operating_revenue | 经营收入 |
| operating_expense_before_deducting_impairment | 营业支出-扣除减值前 |
| profit_after_tax | 除税后溢利 |
| profit_before_tax_income | 除税前溢利 |
| net_income_from_securities_trading_and_investment | 证券交易及投资净收入 |
| net_income_from_foreign_exchange_trading | 外汇交易净收入 |
| interest_income | 利息收入 |
| interest_expense | 利息支出 |
| attributable_profit_to_associated_company | 应占联营公司溢利 |
| other_operating_income_items | 经营收入其他项目 |
| net_service_fee_income | 净服务费收入 |
| service_fee_revenue | 服务费收入 |
| service_fee_expense | 服务费支出 |
| taxation | 税项 |

**资产负债表**

| 字段 | 释义、备注 |
| :--- | :--- |
| revaluation_reserve | 重估储备 |
| fixed_assets | 固定资产 |
| borrowings_from_central_banks | 向中央银行借款 |
| derivative_financial_liabilities | 衍生金融负债 |
| derivative_financial_assets | 衍生金融资产 |
| issued_bonds | 发行债券 |

**现金流量表**

| 字段 | 释义、备注 |
| :--- | :--- |
| cash_paid_to_acquire_investment | 投资支付现金 |
| net_cash_from_investment_business | 投资业务现金净额 |
| sell_fixed_assets | 出售固定资产 |
| profit_from_selling_other_assets | 出售其他资产损（益） |
| cash_received_from_disposal_of_investment | 收回投资所得现金 |
| acquisition_of_subcompany | 收购附属公司 |
| other_impairment_and_provision_cashflow | 其他减值与拨备 |
| other_bank_operation_assets_change_items | 银行—经营资产变动其他项目 |
| other_operation_business_items | 经营业务其他项目 |
| bank_borrowings_from_central_banks_change | 银行—向中央银行借款增(减) |
| net_cash_from_financing_business | 融资业务现金净额 |
| dividend_income_adjustment | 股息（收入）-调整 |
| paid_financing_dividend | 已付股息(融资) |
| profit_before_tax_cashflow | 除税前溢利(业务利润) |

##### 非中国会计准则_保险公司

**利润表**

| 字段 | 释义、备注 |
| :--- | :--- |
| exchange_profit_expense | 汇兑损益-支出 |
| other_income_items | 收入其他项目 |
| ga_expense | 管理费用 |
| total_premium_income | 总保费收入 |
| financing_expense | 财务费用-支出 |
| other_expense | 其他支出 |
| taxation | 税项 |
| other_operating_expense_items | 支出其他项目 |
| investment_income | 投资收益 |
| minority_profit | 少数股东损益 |
| payment_compensation_and_total_expenses | 给付、赔付及费用总计 |
| banks_income | 银行业务收入 |
| profit_after_tax | 除税后溢利 |
| profit_before_tax_income | 除税前溢利 |
| total_income | 收入合计 |
| attributable_profit | 股东应占溢利 |
| basic_earnings_per_share | 每股基本盈利 |

**资产负债表**

| 字段 | 释义、备注 |
| :--- | :--- |
| cash_and_equivalents | 现金及现金等价物 |
| loan | 贷款 |
| insurance_contract | 保险合同 |
| refundable_capital_deposits | 存出资本保证金 |
| special_asset_projects | 特殊资产项目 |
| equity | 权益 |
| fixed_assets | 固定资产 |
| liabilities_for_other_items | 负债其他项目 |

**现金流量表**

| 字段 | 释义、备注 |
| :--- | :--- |
| cash_received_from_disposal_of_investment | 收回投资所得现金 |
| sell_fixed_assets | 出售固定资产 |
| sell_subsidiary_company | 出售附属公司 |
| net_cash_from_investment_business | 投资业务现金净额 |
| cash_paid_to_acquire_investment | 投资支付现金 |
| paid_financing_dividend | 已付股息(融资) |
| paid_financing_interest | 已付利息(融资) |
| absorb_investment_income | 吸收投资所得 |
| issuance_fee_and_expense_for_redeeming_security | 发行费用及赎回证券支出 |
| purchase_fixed_assets | 购买固定资产 |
| other_financing_business_items | 融资业务其他项目 |
| other_investment_business_items | 投资业务其他项目 |
| received_investment_interest | 已收利息—投资 |
| received_investment_dividend | 已收股息—投资 |
| newly_added_loan | 新增借款 |

##### 非中国会计准则_非金融非保险公司

**利润表**

| 字段 | 释义、备注 |
| :--- | :--- |
| turnover | 营业额 |
| operating_revenue | 营运收入 |
| operating_profit | 经营溢利 |
| profit_after_tax | 除税后溢利 |
| profit_before_tax_income | 除税前溢利 |
| financing_cost | 融资成本 |
| salary_and_welfare_expenses | 薪金福利支出 |
| other_expense | 其他支出 |
| minority_profit | 少数股东损益 |
| attributable_profit | 股东应占溢利 |
| sales_and_distribution_expense | 销售及分销费用 |

**资产负债表**

| 字段 | 释义、备注 |
| :--- | :--- |
| current_financial_leasing_liabilities | 流动融资租赁负债 |
| total_equity_and_total_liabilities | 权益总额及负债总额 |
| deferred_tax_liabilities | 递延税项负债 |
| payable_taxes | 应付税项 |
| deferred_tax_assets | 递延税项资产 |
| prepaid_and_receivable_taxes | 预付及应收税项 |

**现金流量表**

| 字段 | 释义、备注 |
| :--- | :--- |
| deposit_change | 存款减少(增加) |
| cash_paid_to_acquire_investment | 投资支付现金 |
| purchase_fixed_assets | 购买固定资产 |
| cash_paid_for_intangible_assets_and_other_assets | 购建无形资产及其他资产 |
| depreciation | 折旧 |
| issuing_stock | 发行股份 |
| received_investment_interest | 已收利息—投资 |
| other_financing_business_items | 融资业务其他项目 |
| received_investment_dividend | 已收股息—投资 |
| paid_financing_interest | 已付利息(融资) |
| paid_financing_dividend | 已付股息(融资) |
| exchange_rate_impact | 汇率影响 |
| cash_received_from_disposal_of_investment | 收回投资所得现金 |
| financial_expense | 财务费用 |
| net_cash_from_financing_business | 融资业务现金净额 |
| other_operation_adjustment_items | 经营调整其他项目 |
| begin_period_cash | 期初现金 |

#### 范例

获取 00001.XHKG 2023 年各报告期所有记录

```python
[In]
get_pit_financials_ex(fields=['turnover','begin_period_cash'], start_quarter='2023q1', end_quarter='2023q4', order_book_ids='00001.XHKG')
[Out]
               info_date      turnover  fiscal_year           standard  if_adjusted
order_book_id quarter
00001.XHKG    2023q2  2023-08-03  1.333770e+11   2023-12-31  非中国会计准则_非金融非保险公司            0
              2023q2  2024-08-15  1.333770e+11   2023-12-31  非中国会计准则_非金融非保险公司            1
              2023q4  2024-03-21  2.755750e+11   2023-12-31  非中国会计准则_非金融非保险公司            0
              2023q4  2025-03-20  2.755750e+11   2023-12-31  非中国会计准则_非金融非保险公司            1
```

获取 00001.XHKG 2023 年查询日期为 20240321 的记录

```python
[In]
get_pit_financials_ex(fields=['turnover','begin_period_cash'], start_quarter='2023q1', end_quarter='2023q4', order_book_ids='00001.XHKG', date='20240321')
[Out]
               info_date      turnover  fiscal_year           standard  if_adjusted
order_book_id quarter
00001.XHKG    2023q2  2023-08-03  1.333770e+11   2023-12-31  非中国会计准则_非金融非保险公司            0
              2023q4  2024-03-21  2.755750e+11   2023-12-31  非中国会计准则_非金融非保险公司            0
```

获取股票列表 2024q3-2024q4 各报告期最新一次记录

```python
[In]
get_pit_financials_ex(fields=['intangible_assets','revenue'], start_quarter='2024q3', end_quarter='2024q4', order_book_ids=['00038.XHKG', '00763.XHKG'])
[Out]
                info_date intangible_assets fiscal_year      standard if_adjusted revenue
order_book_id quarter
00038.XHKG    2024q3  2024-10-29      7.361137e+08   2024-12-31  中国会计准则           0  1.39...
              2024q4  2025-03-27      7.233467e+08   2024-12-31  中国会计准则           0  1.92...
00763.XHKG    2024q3  2024-10-21      8.270417e+09   2024-12-31  中国会计准则           0  9.00...
              2024q4  2025-02-28      7.634037e+09   2024-12-31  中国会计准则           0  1.24...
```

---

### hk.get_detailed_financial_items - 查询财务细分项目(point-in-time 形式)

```python
rqdatac.hk.get_detailed_financial_items(order_book_ids, fields, start_quarter, end_quarter, date=None, statements='latest', market='hk')
```

> **注意事项**
>
> 请先单独安装 rqdatac_hk，导入后使用

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | **必填参数**，合约代码，可传入 order_book_id, order_book_id list ，该参数必填 |
| fields | *list* | **必填参数**，需要返回的财务字段。支持的字段仅限**利润表、资产负债表、现金流量表三大表字段**，具体字段请看 [get_pit_financials_ex](#rqdata-API-financials_hk) 介绍。 |
| start_quarter | *str* | **必填参数**，财报回溯查询的起始报告期，例如'2015q2'代表 2015 年半年报， 该参数必填 。 |
| end_quarter | *str* | **必填参数**，财报回溯查询的截止报告期，例如'2015q4'代表 2015 年年报，该参数必填。 |
| date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 查询日期，默认查询日期为当前最新日期 |
| statements | *str* | 基于查询日期，返回某一个报告期的所有记录或最新一条记录，设置 statements 为 all 时返回所有记录，statements 等于 latest 时返回最新的一条记录，默认为 latest. |
| market | *str* | 市场，仅限'hk'香港市场 |

#### 返回

*pandas DataFrame*

| 固定字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| quarter | *str* | 报告期 |
| info_date | *pandas.Timestamp* | 公告发布日 |
| field | *list* | 需要返回的财务字段。需要返回的财务字段。支持的字段仅限利润表、资产负债表、现金流量表三大表字段，具体字段请看 [get_pit_financials_ex](#get_pit_financials_ex---查询季度财务信息point-in-time-形式) 介绍。 |
| if_adjusted | *int* | 是否为非当期财报数据, 0 代表当期，1 代表非当期（比如 18 年的财报会披露本期和上年同期的数值，17 年年报的财务数值在 18 年年报中披露的记录则为非当期， 17 年年报的财务数值在 17 年年报中披露则为当期）。 |
| fiscal_year | *pandas.Timestamp* | 财政年度 |
| standard | *str* | 会计准则，中国会计准则、非中国会计准则_金融公司、非中国会计准则_保险公司、非中国会计准则_非金融非保险公司 |
| relationship | *int* | 运算符号，0 表示其中项不参与计算，1 表示正号，-1 表示负号。<br>资产负债表都是按照正值， relationship 均为 1，利润表和现金流量表区分正负值。 |
| subject | *str* | fields 下面所有细分项目名称（实际出现在财务报表中的名称） |
| amount | *float* | 未做港币汇率转换的原始值（实际出现在财务报表中的原始值） |
| currency | *str* | 货币单位 |

#### 范例

获取 02318.XHKG 2023q1-2023q2 fields 下所有细分项目的最新一次记录

```python
[In]
import rqdatac
import rqdatac_hk
rqdatac.init()
rqdatac.hk.get_detailed_financial_items(order_book_ids=['02318.XHKG'],start_quarter='2023q1', end_quarter='2023q2', fields=['other_operating_expense_items'])
[Out]
                     info_date fiscal_year                          field  relationship  amount currency                          subject ...
order_book_id quarter
02318.XHKG    2023q1 2024-04-23  2023-12-31  other_operating_expense_items           1.0  -4.600      CNY            其他业务成本 ...
              2023q1 2024-04-23  2023-12-31  other_operating_expense_items           1.0  -2.634      CNY            资产处置损失 ...
              2023q1 2024-04-23  2023-12-31  other_operating_expense_items           1.0  -1.894      CNY            其他收益 ...
              2023q2 2024-08-22  2023-12-31  other_operating_expense_items           1.0  -1.440      CNY            税金及附加 ...
              2023q2 2024-08-22  2023-12-31  other_operating_expense_items           1.0  -5.329      CNY            其他业务成本 ...
              2023q2 2024-08-22  2023-12-31  other_operating_expense_items           1.0  -4.368      CNY            资产处置损失 ...
```

---

## 港股因子数据

### get_factor - 获取因子值

```python
get_factor(order_book_ids, factor, start_date=None, end_date=None, universe=None, expect_df=True, market='cn')
```

默认返回指定因子上一个交易日的值，目前港股因子仅支持获取市值和流通市值。

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* or *str list* | **必填参数**，合约代码，可传入 order_book_id, order_book_id list |
| factor | *str* or *str list* | **必填参数**，因子名称，见下方，也可查询 get_all_factor_names(market='hk') 得到所有有效因子字段 |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期。注：如使用开始日期，则必填结束日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期。注：若使用结束日期，则开始日期必填 |
| expect_df | *boolean* | 默认返回 pandas dataframe。如果调为 False，则返回 原有的数据结构 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*pandas DataFrame*

**factor 支持因子：**

| 字段 | 中文名 | 说明 | 公式 |
| :--- | :--- | :--- | :--- |
| hk_share_market_val | 港股市值 | 港股市值 = 已上市港股股数 * 港股未复权收盘价 此处股本采用 PIT 处理方式 | total_hk * close |
| hk_share_market_val_in_circulation | 港股流通市值 | 港股流通市值 = 可在港股交易的股数 * 港股未复权收盘价 此处股本采用 PIT 处理方式 | total_hk1 * close |
| hk_total_market_val | 港股总市值 | 总市值 = 总股本 * 港股未复权收盘价 此处股本采用 PIT 处理方式 | total * close |

#### 范例

获取单支港股市值数据：

```python
[In]
get_factor('00020.XHKG',['hk_share_market_val','hk_share_market_val_in_circulation'], start_date=20250804, end_date=20250808, market='hk')
[Out]
                     hk_share_market_val  hk_share_market_val_in_circulati
order_book_id date
00020.XHKG    2025-08-04    6.187846e+10                      6.089601e+10
              2025-08-05    6.226520e+10                      6.127661e+10
              2025-08-06    6.342542e+10                      6.241841e+10
              2025-08-07    6.342542e+10                      6.241841e+10
              2025-08-08    6.265194e+10                      6.165721e+10
```

获取多支港股市值数据：

```python
[In]
get_factor(['00020.XHKG','03750.XHKG'],['hk_share_market_val','hk_share_market_val_in_circulation'], start_date=20250804, end_date=20250805, market='hk')
[Out]
                     hk_share_market_val  hk_share_market_val_in_circulati
order_book_id date
00020.XHKG    2025-08-04    6.187846e+10                      6.089601e+10
              2025-08-05    6.226520e+10                      6.127661e+10
03750.XHKG    2025-08-04    6.507905e+10                      6.507905e+10
              2025-08-05    6.423710e+10                      6.423710e+10
```

---

### get_all_factor_names - 获取因子字段列表

```python
get_all_factor_names(type=None, market='cn')
```

目前港股因子仅支持获取市值和流通市值

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| type | *str* | 默认返回所有因子<br>'eod_indicator'：估值有关指标 |
| market | *str* | 默认是中国内地市场('cn') 。可选'cn' - 中国内地市场；'hk' - 香港市场 |

#### 返回

*list*

#### 范例

获取市值和流通市值因子

```python
[In]
get_all_factor_names(type='eod_indicator',market='hk')
[Out]
['hk_share_market_val', 'hk_share_market_val_in_circulation','hk_total_market_val']
```

---

## 港股公告相关

### hk.get_announcement - 获取港股公告数据

```python
rqdatac.hk.get_announcement(order_book_ids, start_date=None, end_date=None, fields=None, market='hk')
```

> **注意事项**
>
> 请先单独安装 rqdatac_hk，导入后使用

#### 参数

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str or list* | **必填参数**，合约代码，给出单个或多个 order_book_id |
| start_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 开始日期。注：如使用开始日期，则必填结束日期 |
| end_date | *int, str, datetime.date, datetime.datetime, pandas.Timestamp* | 结束日期。注：若使用结束日期，则开始日期必填 |
| fields | *list* | 可选字段见下方返回，若不指定，则默认获取所有字段 |
| market | *str* | 市场，仅限'hk'香港市场 |

#### 返回

*pandas DataFrame*

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| order_book_ids | *str* | 合约代码 |
| info_date | *pandas.Timestamp* | 发布日期 |
| meida | *str* | 媒体出处 |
| title | *str* | 标题 |
| language | *str* | 语言 |
| file_type | *str* | 文件格式 |
| announcement_link | *str* | 公告链接 |
| first_category | *str* | 一级英文公告分类，中英文公告分类映射，可点此[中英文映射表](https://www.ricequant.com/doc/rqdata-python/pdf/%E6%B8%AF%E8%82%A1%E5%85%AC%E5%91%8A%E5%88%86%E7%B1%BB%E4%B8%AD%E8%8B%B1%E6%96%87%E6%98%A0%E5%B0%84%E8%A1%A8.xlsx)下载查看，下同。 |
| second_category | *str* | 二级英文公告分类 |
| third_category | *str* | 三级英文公告分类 |

#### 范例

获取一个合约某个时间段内的公司公告数据

```python
[In]
import rqdatac
import rqdatac_hk
rqdatac.hk.get_announcement(order_book_ids=['00638.XHKG'],start_date=20251127,end_date=20251128)
[Out]
                                          media                                       title language file_type ...
order_book_id  info_date
00638.XHKG     2025-11-28  香港交易所  广和通(00638)公告及通告 - [海外监管公告-董事会/监事会...  ...      ...
               2025-11-28  香港交易所  广和通(00638)公告及通告 - [海外监管公告-董事会/监事会...  ...      ...
               2025-11-28  香港交易所  广和通(00638)公告及通告 - [海外监管公告-其他]海外监管...  ...      ...
               ...         ...    ...                                   ...      ...
               2025-11-28  香港交易所  FIBOCOM(00638)Announcements and Notices - [Ter..  ...      ...
               2025-11-28  香港交易所  FIBOCOM(00638)Announcements and Notices - [Ter..  ...      ...
               2025-11-28  香港交易所  FIBOCOM(00638)Announcements and Notices - [Ter..  ...      ...
```