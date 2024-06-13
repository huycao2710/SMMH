import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Đường dẫn tới file CSV
file_path = "smmh.csv"
#Đọc file CSV vào Dataframe
smmh = pd.read_csv(file_path)

#Cài đặt để hiển thị tất cả các cột 
# pd.set_option("display.max_columns", None)

#In 5 giá trị đầu của tập dữ liệu
print(smmh.head())

#Xác định số dòng và số cột trong dataset
# print(smmh.shape)
# (481, 21)

#Thay đổi feature cho thông tin cần sử dụng
smmh.rename(columns = {'1. What is your age?':'Age','2. Gender':'Gender','3. Relationship Status':'Relationship Status',
                       '4. Occupation Status':'Occupation',
                       '5. What type of organizations are you affiliated with?':'Affiliations',
                       '6. Do you use social media?':'Social Media User?',
                       '7. What social media platforms do you commonly use?':'Platforms Used',
                       '8. What is the average time you spend on social media every day?':'Hours Per Day',
                       '9. How often do you find yourself using Social media without a specific purpose?':'ADHD Q1',
                       '10. How often do you get distracted by Social media when you are busy doing something?':'ADHD Q2',
                       "11. Do you feel restless if you haven't used Social media in a while?":'Anxiety Q1',
                       '12. On a scale of 1 to 5, how easily distracted are you?':'ADHD Q3',
                       '13. On a scale of 1 to 5, how much are you bothered by worries?':'Anxiety Q2',
                       '14. Do you find it difficult to concentrate on things?':'ADHD Q4',
                       '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?':'Self Esteem Q1',
                       '16. Following the previous question, how do you feel about these comparisons, generally speaking?':'Self Esteem Q2',
                       '17. How often do you look to seek validation from features of social media?':'Self Esteem Q3',
                       '18. How often do you feel depressed or down?':'Depression Q1',
                       '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?':'Depression Q2',
                       '20. On a scale of 1 to 5, how often do you face issues regarding sleep?':'Depression Q3' },inplace=True)
#Affiliations:
#ADHD_Attention Deficit Hyperractivity Disorder: Rối loạn chú ý và hoạt động quá động
#Anxiety: Lo âu
#Self Esteem: Câu hỏi về tự trọng
#Depression: Trầm cảm

#check lại các 
features = list(smmh.columns)
# print(features)
#['Timestamp', 
# 'Age', 
# 'Gender', 
# 'Relationship Status', 
# 'Occupation', 
# 'Affiliations', 
# 'Social Media User?', 
# 'Platforms Used', 
# 'Hours Per Day', 
# 'ADHD Q1', 
# 'ADHD Q2', 
# 'Anxiety Q1', 
# 'ADHD Q3', 
# 'Anxiety Q2', 
# 'ADHD Q4', 
# 'Self Esteem Q1', 
# 'Self Esteem Q2', 
# 'Self Esteem Q3', 
# 'Depression Q1', 
# 'Depression Q2', 
# 'Depression Q3']

#Sắp xếp lại thứ tự câu hỏi về ADHD và Axiety để liên tục
features[11], features[12] = features[12], features[11]
features[12], features[14] = features[14], features[12]
features[13], features[14] = features[14], features[13]
smmh = smmh[features]
# print("Features trước khi loại bỏ:")
# print(features)
#['Timestamp', 
# 'Age', 
# 'Gender', 
# 'Relationship Status', 
# 'Occupation', 
# 'Affiliations', 
# 'Social Media User?', 
# 'Platforms Used', 
# 'Hours Per Day', 
# 'ADHD Q1', 
# 'ADHD Q2', 
# 'ADHD Q3', 
# 'ADHD Q4', 
# 'Anxiety Q1', 
# 'Anxiety Q2', 
# 'Self Esteem Q1', 
# 'Self Esteem Q2', 
# 'Self Esteem Q3', 
# 'Depression Q1', 
# 'Depression Q2', 
# 'Depression Q3']

#Loại bỏ features không sử dụng tới
to_drop = ['Timestamp',
          'Affiliations']

smmh.drop(to_drop, inplace=True, axis=1)

features = list(smmh.columns)
# print("Features sau khi loại bỏ:")
print(features)
#['Age', 
# 'Gender', 
# 'Relationship Status', 
# 'Occupation', 
# 'Social Media User?', 
# 'Platforms Used', 
# 'Hours Per Day', 
# 'ADHD Q1', 
# 'ADHD Q2', 
# 'ADHD Q3', 
# 'ADHD Q4', 
# 'Anxiety Q1', 
# 'Anxiety Q2', 
# 'Self Esteem Q1', 
# 'Self Esteem Q2', 
# 'Self Esteem Q3', 
# 'Depression Q1', 
# 'Depression Q2', 
# 'Depression Q3']

#Xác định số dòng và số cột sau khi loại bỏ features trong dataset
# print(smmh.shape)
# (481, 19)

#Xử lý giá trị thiếu: Blank Values/NaN/null
missing_values = smmh.isna().sum()
# print(missing_values)
# Age                    0
# Gender                 0
# Relationship Status    0
# Occupation             0
# Social Media User?     0
# Platforms Used         0
# Hours Per Day          0
# ADHD Q1                0
# ADHD Q2                0
# ADHD Q3                0
# ADHD Q4                0
# Anxiety Q1             0
# Anxiety Q2             0
# Self Esteem Q1         0
# Self Esteem Q2         0
# Self Esteem Q3         0
# Depression Q1          0
# Depression Q2          0
# Depression Q3          0
# dtype: int64

#Kiểm tra giá trị rỗng và kiểm tra số lượng bản ghi của mỗi cột feature
# smmh.info()

#Đồng bộ hoá giá trị Non-Binary trong Gender
# Genders = set(smmh['Gender'])
# print("Genders trước khi đồng bộ:")
# print(Genders)
# {'Female', 'There are others???', 'unsure ', 'Non-binary', 'Trans', 'NB', 'Male', 'Non binary ', 'Nonbinary '}

#Thay thế để đồng bộ giá trị Non-Binary
smmh.replace('Non-binary','Non-Binary', inplace=True)
smmh.replace('Nonbinary ','Non-Binary', inplace=True)
smmh.replace('NB','Non-Binary', inplace=True)
smmh.replace('Non binary ','Non-Binary', inplace=True)

#Sau khi đồng bộ giá trị Non-Binary trong Gender
# Genders = set(smmh['Gender'])
# print("Genders sau khi đồng bộ:")
# print(Genders)
#{'Female', 'Male', 'unsure ', 'There are others???', 'Non-Binary', 'Trans'}

#Đồng bộ hoá Age về kiểu int64
smmh['Age'] = smmh['Age'].astype('int64')

# print(smmh['Age'].dtype)
# smmh.info()

#Điều chỉnh lại thứ tự thang điểm với Self Esteem Q2_"Following the previous question, how do you feel about these comparisons, generally speaking?"
# 5 = very negative - tiêu cực
# 4 = slightly negative - hơi tiêu cực
# 3 = neutral - trung lập
# 2 = slightly positive - hơi tích cực
# 1 = very positive - tích cực
smmh.loc[smmh['Self Esteem Q2'] == 1, 'Self Esteem Q2'] = 5
smmh.loc[smmh['Self Esteem Q2'] == 2, 'Self Esteem Q2'] = 4
smmh.loc[smmh['Self Esteem Q2'] == 3, 'Self Esteem Q2'] = 3
smmh.loc[smmh['Self Esteem Q2'] == 4, 'Self Esteem Q2'] = 2
smmh.loc[smmh['Self Esteem Q2'] == 5, 'Self Esteem Q2'] = 1

#Tổng hợp điểm về 4 loại sàng lọc
#4 câu về ADHD: max 20
#2 câu về Anxiety: max 10
#3 câu về Self Esteem: max 15
#3 câu về Depression: max 15
#Với 12 câu hỏi sàng lọc thì tổng điểm max 60
ADHD = ['ADHD Q1', 'ADHD Q2', 'ADHD Q3', 'ADHD Q4']
smmh['ADHD Score'] = smmh[ADHD].sum(axis=1)

Anxiety = ['Anxiety Q1', 'Anxiety Q2']
smmh['Anxiety Score'] = smmh[Anxiety].sum(axis=1)

SelfEsteem = ['Self Esteem Q1', 'Self Esteem Q2', 'Self Esteem Q3']
smmh['Self Esteem Score'] = smmh[SelfEsteem].sum(axis=1)

Depression = ['Depression Q1', 'Depression Q2', 'Depression Q3']
smmh['Depression Score'] = smmh[Depression].sum(axis=1)

Total = ['ADHD Score', 'Anxiety Score', 'Self Esteem Score', 'Depression Score']
smmh['Total Score'] = smmh[Total].sum(axis=1)

smmh.drop(columns=ADHD + Anxiety + SelfEsteem + Depression, inplace=True)

# print(smmh.head(5))

#Tối giản giá trị trong cột Hours Per Day
smmh.loc[smmh['Hours Per Day'] == 'More than 5 hours', 'Hours Per Day'] = '5.5h'
smmh.loc[smmh['Hours Per Day'] == 'Between 2 and 3 hours', 'Hours Per Day'] = '2.5h'
smmh.loc[smmh['Hours Per Day'] == 'Between 3 and 4 hours', 'Hours Per Day'] = '3.5h'
smmh.loc[smmh['Hours Per Day'] == 'Between 1 and 2 hours', 'Hours Per Day'] = '1.5h'
smmh.loc[smmh['Hours Per Day'] == 'Between 4 and 5 hours', 'Hours Per Day'] = '4.5h'
smmh.loc[smmh['Hours Per Day'] == 'Less than an Hour', 'Hours Per Day'] = '0.5h'
# print(smmh.head(5))

#Chỉnh lại câu trả lời có sử dụng mạng xã hội
#Kiểm tra tổng thể
social_media_user_counts = smmh['Social Media User?'].value_counts()
# print("Counts of Social Media User?:")
# print(social_media_user_counts)

#Tìm giá trị No
no_answers = smmh[smmh['Social Media User?'] == 'No']
# print("Rows with 'No' answers:")
# print(no_answers)

#Thay thế No thành Yes
smmh.loc[smmh['Social Media User?'] == 'No', 'Social Media User?'] = 'Yes'
social_media_user_counts = smmh['Social Media User?'].value_counts()

# print("Counts of Social Media User?:")
# print(social_media_user_counts)

#Kiểm tra cột Age thì có giá trị bất thường
Age_counts = smmh['Age'].value_counts()
Age_counts = Age_counts.sort_index(ascending=True)
# print("Counts Age (Ascending Order):")
# print(Age_counts)

age_answers = smmh[smmh['Age'] == 91]
# print("Rows with Age equal to 91:")
# print(age_answers)

smmh.loc[smmh['Age'] == 91, 'Age'] = 19
# print("Row 256:")
# print(smmh.iloc[256])

#Chuyển đổi các Platforms Used thành 3 loại 
# Social Networking (Facebook,Twitter, Discord, TikTok) - SN
# Media Sharing (Pinterest, Youtube, Snapchat, Instagram) - MS
# Discussion Forum (Reddit) - DF
def categorize_platforms(platforms_used):
    categories = []
    if any(platform in platforms_used for platform in ['Facebook', 'Twitter', 'Discord', 'TikTok']):
        categories.append('SN')
    if any(platform in platforms_used for platform in ['Pinterest', 'YouTube', 'Snapchat', 'Instagram']):
        categories.append('MS')
    if 'Reddit' in platforms_used:
        categories.append('DF')
    return ', '.join(categories) if categories else 'Other/Unknown'

smmh['Platforms'] = smmh['Platforms Used'].apply(categorize_platforms)

smmh.drop(columns=['Platforms Used'], inplace=True)

print(smmh)

# output_file_path = "tienxulydulieu_smmh.csv"

# # Xuất DataFrame về tệp CSV
# smmh.to_csv(output_file_path, index=False)

# print(f"Đã xuất dữ liệu đã tiền xử lý vào file CSV: {output_file_path}")
