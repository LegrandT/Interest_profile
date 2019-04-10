import xlrd
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import statistics


from IPython.display import display
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns',100)

# file_location1 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Opens.csv"
# odfOpen = pd.read_csv(file_location1)
# file_location2 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Sends.csv"
# odfSends = pd.read_csv(file_location2)
file_location3 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Clicks.csv"
odfClicks = pd.read_csv(file_location3, encoding= 'ISO-8859-1')
file_location4 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Customer_data.csv"
odfCosDat = pd.read_csv(file_location4)
file_location5 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\newsletter_link_tags_def.csv"
odfNewLiTaDe = pd.read_csv(file_location5)
file_location6 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\newsletter_link_tags_0817.csv"
odfNewLiTa08 = pd.read_csv(file_location6)


# df = odfCosDat[['KLTID', 'INCOME_PARTTIME_LATEST_JOB', 'OMS_GESLACHT']].copy()
#
# df['INCOME_PARTTIME_LATEST_JOB'] = [float(str(p).replace(",", ".")) for p in df['INCOME_PARTTIME_LATEST_JOB']]
# df['OMS_GESLACHT'] = [0 if p == 'Man' else 1 if p == 'Vrouw' else 0.5 for p in df['OMS_GESLACHT']]

# get all kltid with data about them
df = odfCosDat[['KLTID']].copy()


# get info about GIFSF;MD;OB;DZ;PK of the article links
new = odfNewLiTaDe["LINK_ID;LINK_URL;GIFSF;MD;OB;DZ;PK"].str.split(";", expand = True)
temp = odfNewLiTaDe
temp.drop(columns=["LINK_ID;LINK_URL;GIFSF;MD;OB;DZ;PK"], inplace=True)
temp['LINK_ID'] = new[0].astype(int)
temp['GIFSF'] = new[2]
temp['MD'] = new[3]
temp['OB'] = new[4]
temp['DZ'] = new[5]
temp['PK'] = new[6]
link = pd.merge(temp, odfNewLiTa08, on=['LINK_ID'], how='outer')
link.drop(["URL"], axis=1, inplace=True)



for i in range(len(link)):
    if link['ARTICLE_TAG'][i] == 'GIFSF':
        link.loc[i, 'GIFSF'] = 1
    elif link['ARTICLE_TAG'][i] == 'MD':
        link.loc[i, 'MD'] = 1
    elif link['ARTICLE_TAG'][i] == 'OB':
        link.loc[i, 'OB'] = 1
    elif link['ARTICLE_TAG'][i] == 'DZ':
        link.loc[i, 'DZ'] = 1
    elif link['ARTICLE_TAG'][i] == 'PK':
        link.loc[i, 'PK'] = 1
link.drop("ARTICLE_TAG", axis=1, inplace=True)
link.replace(['', np.nan], 0, inplace=True)
link.set_index('LINK_ID', inplace=True)
link = link.astype(int)


cli = odfClicks.copy()
# associate all clicks with the GIFSF;MD;OB;DZ;PK of the article
cli = pd.merge(cli, link, on=['LINK_ID'], how='left')
cli.drop(["ARTICLE_TAG", 'JOB_ID', 'NS_LINKNAME', 'LINK_ID', 'CLICK_TIME'], axis=1, inplace=True)
cli.dropna(inplace=True)
cli = cli.groupby(['KLTID']).max()
tempcli = cli = cli.groupby(['KLTID']).sum()
tempcli['sum'] = tempcli[['GIFSF', 'MD', 'OB', 'DZ', 'PK']].sum(axis=1)
cli['fGIFSF'] = cli['GIFSF']/tempcli['sum']
cli['fMD'] = cli['MD']/tempcli['sum']
cli['fOB'] = cli['OB']/tempcli['sum']
cli['fDZ'] = cli['DZ']/tempcli['sum']
cli['fPK'] = cli['PK']/tempcli['sum']
cli.replace(np.nan, 0, inplace=True)


# keep only the info when we have cosdata info

df = pd.merge(df, cli, on=['KLTID'], how='left')
df.dropna(inplace=True)
df.set_index('KLTID', inplace=True)

df = pd.merge(df, odfCosDat.drop(['DTM_EVT', 'CD_PLAATS', 'LAST_DIVORCE_DATE'
, 'INCOMECLASS_PARTTIME_TOT_YEAR', 'PARTTIME_FACTOR_TOT'
, 'PARTTIME_FACTOR_LATEST_JOB'
, 'INCOMECLASS_PARTTIME_LAT_JOB', 'DTM_OVL', 'DECEASED_YN'], axis=1), on=['KLTID'], how='left')
# full set is : ['KLTID', 'DTM_EVT', 'OMS_GESLACHT', 'LEEFT', 'OMS_SMLVRM', 'CD_PLAATS', 'CD_LAND', 'CD_STATUS_MCD', 'COMMUNICATIONCHOICE'
# , 'NEWSLETTER', 'EVER_DIVORCED', 'LAST_DIVORCE_DATE', 'NUMBER_OF_DIVORCES', 'PARTNER_AT_ABP', 'NUMBER_OF_DLN', 'PARTTIME_INCOME_TOT'
# , 'INCOMECLASS_PARTTIME_TOT_YEAR', 'PARTTIME_FACTOR_TOT', 'PARTTIME_FACTOR_TOTAAL_CLASS', 'TYPE_DLN_BIGGEST_JOB', 'SECTOR_MMS_BIGGEST_JOB'
# , 'SECTOR_GWS_BIGGEST_JOB', 'NETTOPENSION_IND', 'PARTTIME_FACTOR_LATEST_JOB', 'PARTTIME_FACTOR_CLASS_LAT_JOB', 'INCOME_PARTTIME_LATEST_JOB'
# , 'INCOMECLASS_PARTTIME_LAT_JOB', 'SECTOR_MMS_LATEST_JOB', 'SECTOR_GWS_LATEST_JOB', 'DTM_OVL', 'DECEASED_YN']

# df = pd.merge(df, odfCosDat[['KLTID', 'NUMBER_OF_DIVORCES', 'PARTTIME_INCOME_TOT', 'OMS_GESLACHT', 'LEEFT', 'OMS_SMLVRM']], on=['KLTID'], how='left')


df['NUMBER_OF_DIVORCES'].replace(np.nan, 0, inplace=True)

df['OMS_GESLACHT'] = [0 if p == 'Man' else 1 if p == 'Vrouw' else 0.5 for p in df['OMS_GESLACHT']]

df['PARTTIME_INCOME_TOT'] = [float(str(p).replace(",", ".")) for p in df['PARTTIME_INCOME_TOT']]

df['NETTOPENSION_IND'] = [0 if p == 'N' else 1 if p == 'Y' else 0.5 for p in df['NETTOPENSION_IND']]

# df['PARTTIME_FACTOR_TOT'] = [float('0'.join(str(p).replace(",", "."))) for p in df['PARTTIME_FACTOR_TOT']]

# df['CD_PLAATS'] = [str(p) for p in df['CD_PLAATS']]
#
# df['CD_LAND'] = [str(p) for p in df['CD_LAND']]

df['EVER_DIVORCED'] = [0 if p == 'N' else 1 if p == 'Y' else 0.5 for p in df['EVER_DIVORCED']]

df['PARTNER_AT_ABP'] = [0 if p == 'N' else 1 if p == 'Y' else 0.5 for p in df['PARTNER_AT_ABP']]

# CD_STATUS_MCD

df['NEWSLETTER'] = [0 if p == 'N' else 1 if p == 'Y' else 0.5 for p in df['NEWSLETTER']]

df['COMMUNICATIONCHOICE'] = [0 if p == 'Paper' else 1 if p == 'Digital' else 0.5 for p in df['COMMUNICATIONCHOICE']]

df['INCOME_PARTTIME_LATEST_JOB'] = [float(str(p).replace(",", ".")) for p in df['INCOME_PARTTIME_LATEST_JOB']]

# tempmean = sum(df['PARTTIME_INCOME_TOT'].fillna(0))/len(df['PARTTIME_INCOME_TOT'])
# tempstd = statistics.stdev(df['PARTTIME_INCOME_TOT'].dropna())
# df['PARTTIME_INCOME_TOT'] = [(p-tempmean)/tempstd for p in df['PARTTIME_INCOME_TOT']]

tempmax = max(df['PARTTIME_INCOME_TOT'])
df['PARTTIME_INCOME_TOT'] = [p/tempmax for p in df['PARTTIME_INCOME_TOT']]

tempmax = max(df['INCOME_PARTTIME_LATEST_JOB'])
df['INCOME_PARTTIME_LATEST_JOB'] = [p/tempmax for p in df['INCOME_PARTTIME_LATEST_JOB']]

df['LEEFT'] = [float(str(p).replace(",", ".")) for p in df['LEEFT']]
tempmax = max(df['LEEFT'])
df['LEEFT'] = [p/tempmax for p in df['LEEFT']]


# df['OMS_SMLVRM'] = [0 if p == 'Gehuwd geweest' else 0 if p == 'Ongehuwd' else 1 if p == 'Gehuwd' else 1 if p == 'Partnerschap bij fonds' else 0.5 for p in df['OMS_SMLVRM']]
# df.drop_duplicates(subset=['OMS_SMLVRM'], keep='first', inplace = True)
df = df.dropna()

# df['EVER_DIVORCED'] = [0 if p == 'N' else 1 for p in df['EVER_DIVORCED']]

# print(df.dtypes)
display(df)
df = pd.get_dummies(df)
display(df)
#
# os.remove('C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\PreprocessedData2.xlsx')
# writer = pd.ExcelWriter('C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\PreprocessedData2.xlsx')
# df.to_excel(writer, 'Sheet1')
# writer.save()
#













# print(df.loc[df['KLTID'] == 1600])
# print(cli.loc[cli['KLTID'] == 1600])
# cli = cli.groupby(['KLTID']).sum()
# display(cli)
# df = df.groupby(['KLTID']).sum()
# display(df)
# print(df.loc[df['KLTID'] == 1594265])
#
# for i in range(len(cli)):
#     print(i, 'out of', len(cli))
#     if cli.iloc[i]['KLTID'] in df.index:
#         # print(int(link.loc[int(cli.iloc[i]['LINK_ID']), 'GIFSF']))
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] + int(cli.iloc[i]['GIFSF'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'MD'] + int(cli.iloc[i]['MD'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'OB'] + int(cli.iloc[i]['OB'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'DZ'] + int(cli.iloc[i]['DZ'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'PK'] + int(cli.iloc[i]['PK'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'Clicked'] = 1
#
#

################################################

# display(df)
#
# for i in range(len(cli)):
#     print(i, 'out of', len(cli))
#     if cli.iloc[i]['KLTID'] in df.index and int(cli.iloc[i]['LINK_ID']) in link.index:
#         # print(int(link.loc[int(cli.iloc[i]['LINK_ID']), 'GIFSF']))
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] + int(link.loc[int(cli.iloc[i]['LINK_ID']), 'GIFSF'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'MD'] + int(link.loc[int(cli.iloc[i]['LINK_ID']), 'MD'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'OB'] + int(link.loc[int(cli.iloc[i]['LINK_ID']), 'OB'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'DZ'] + int(link.loc[int(cli.iloc[i]['LINK_ID']), 'DZ'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'GIFSF'] = df.loc[int(cli.iloc[i]['KLTID']), 'PK'] + int(link.loc[int(cli.iloc[i]['LINK_ID']), 'PK'])
#         df.loc[int(cli.iloc[i]['KLTID']), 'Clicked'] = 1
#
#
#
#
#




# display(link.loc[11353389][:])
# display(df.loc[8041660][:])




# x = []
# y = []
#
# x.append(odfCosDat['KLTID'].values.tolist())
# x.append(odfCosDat['INCOME_PARTTIME_LATEST_JOB'].values.tolist())
# x.append(odfCosDat['OMS_GESLACHT'].values.tolist())
# x[1] = [float(str(p).replace(",", ".")) for p in x[0]]
# maX0 = max(x[0])
# x[1] = [p/maX0 for p in x[1]] # add a weight for more important feature ?
#
# x[2] = [0 if p == 'Man' else 1 if p == 'Vrouw' else 0.5 for p in x[2]]
#
# x = list(map(list, zip(*x)))
#
#
# for i in range(len(x)):
#     y.append([0, 0, 0, 0, 0])
#
#
# temp = odfNewLiTaDe
# new = temp["LINK_ID;LINK_URL;GIFSF;MD;OB;DZ;PK"].str.split(";", expand = True)
# temp.drop(columns=["LINK_ID;LINK_URL;GIFSF;MD;OB;DZ;PK"], inplace=True)
# temp['LINK_ID'] = new[0].astype(int)
# temp['LINK_URL'] = new[1]
# temp['GIFSF'] = new[2]
# temp['MD'] = new[3]
# temp['OB'] = new[4]
# temp['DZ'] = new[5]
# temp['PK'] = new[6]
#
# link = pd.merge(temp, odfNewLiTa08, on=['LINK_ID'], how='outer')
# link.drop(["LINK_URL", "URL"], axis=1, inplace=True)
#
#
# for i in range(len(link)):
#     if link['ARTICLE_TAG'][i] == 'GIFSF':
#         link.loc[i, 'GIFSF'] = 1
#     elif link['ARTICLE_TAG'][i] == 'MD':
#         link.loc[i, 'MD'] = 1
#     elif link['ARTICLE_TAG'][i] == 'OB':
#         link.loc[i, 'OB'] = 1
#     elif link['ARTICLE_TAG'][i] == 'DZ':
#         link.loc[i, 'DZ'] = 1
#     elif link['ARTICLE_TAG'][i] == 'PK':
#         link.loc[i, 'PK'] = 1
# link.drop("ARTICLE_TAG", axis=1, inplace=True)
# link.replace(['', np.nan], 0, inplace=True)
# link.set_index('LINK_ID', inplace=True)
#
# # display(link)
# # print(len(y))
#
# print(len(odfClicks['KLTID']))
# for i in range(len(odfClicks['KLTID'])):
#         if str(odfClicks.loc[i]['LINK_ID']) != 'nan':
#             y[i][0] = y[i][0] + int(link.loc[int(odfClicks.loc[i]['LINK_ID'])]['GIFSF'])
#             y[i][1] = y[i][1] + int(link.loc[int(odfClicks.loc[i]['LINK_ID'])]['MD'])
#             y[i][2] = y[i][2] + int(link.loc[int(odfClicks.loc[i]['LINK_ID'])]['OB'])
#             y[i][3] = y[i][3] + int(link.loc[int(odfClicks.loc[i]['LINK_ID'])]['DZ'])
#             y[i][4] = y[i][4] + int(link.loc[int(odfClicks.loc[i]['LINK_ID'])]['PK'])
# count = len(x)
# i=0
# while i < count:
#     if str(x[i][1]) == 'nan':
#         del x[i]
#         i = i-1
#         count = count-1
#     else:
#         i = i+1
# print(y)

# display(link)
# link = pd.merge(link, odfNewLiTaDe, on=['LINK_ID'], how='left')
# display(link)

# link = link.loc[:, ['KLTID', 'LINK_ID']]
# link.dropna(axis='columns')
# link = link[~link.isin(['NaN', 'NaT']).any(axis=1)]
# # link.drop_duplicates(subset=['PARTNER_AT_ABP'], keep='first', inplace = True)
# print(link)


# # GIFSF;MD;OB;DZ;PK
# for i in range(len(odfClicks['KLTID'])):
#     if odfClicks['LINK_ID'][i] <= 11886031:
#         if odfClicks['ARTICLE_TAG'][i] == 'GIFSF':
#             y[odfClicks['KLTID'][i]][0] = y[odfClicks['KLTID'][i]][0] + 1
#         elif odfClicks['ARTICLE_TAG'][i] == 'MD':
#             y[odfClicks['KLTID'][i]][1] = y[odfClicks['KLTID'][i]][1] + 1
#         elif odfClicks['ARTICLE_TAG'][i] == 'OB':
#             y[odfClicks['KLTID'][i]][2] = y[odfClicks['KLTID'][i]][2] + 1
#         elif odfClicks['ARTICLE_TAG'][i] == 'DZ':
#             y[odfClicks['KLTID'][i]][3] = y[odfClicks['KLTID'][i]][3] + 1
#         elif odfClicks['ARTICLE_TAG'][i] == 'PK':
#             y[odfClicks['KLTID'][i]][4] = y[odfClicks['KLTID'][i]][4] + 1
#     else:
#
