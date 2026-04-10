import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Student Performance Predictor', page_icon='🎓', layout='centered')

@st.cache_data
def load_data():
    df1 = pd.read_csv('student-mat.csv', sep=';')
    df2 = pd.read_csv('student-por.csv', sep=';')
    df = pd.concat([df1, df2], ignore_index=True)
    df.columns = [
        'school', 'sex', 'age', 'address', 'fam_size', 'parents_status', 'mother_edu', 'father_edu',
        'mother_job', 'father_job', 'reason', 'guardian', 'travel_time', 'study_time',
        'failures', 'school_support', 'fam_support', 'paid_courses', 'activities', 'nursery',
        'higher', 'internet', 'romantic', 'fam_rel', 'free_time', 'go_out', 'weekday_alc',
        'weekend_alc', 'health', 'absences', 'G1', 'G2', 'final_grade'
    ]
    df = df.drop_duplicates(ignore_index=True)
    df['remarks'] = np.where(df['final_grade'] <= 11, 'Poor', np.where(df['final_grade'] <= 15, 'Fair', 'Excellent'))
    return df

@st.cache_data
def prepare_features(df):
    dfd = df.copy()
    X = pd.get_dummies(dfd.drop([
        'final_grade', 'remarks', 'sex', 'fam_size', 'parents_status', 'father_job', 'guardian',
        'school_support', 'fam_support', 'paid_courses', 'activities', 'nursery', 'fam_rel',
        'free_time', 'absences'
    ], axis=1))
    y = dfd['remarks']
    return X, y

@st.cache_resource
def train_model(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=10)
    model = RandomForestClassifier(random_state=42)
    model.fit(xtrain, ytrain)
    score = model.score(xtest, ytest)
    return model, score, X.columns


def plot_stacked_bar(df, col, title, colormap):
    out_tab = pd.crosstab(df.remarks, df[col])
    out_perc = out_tab.apply(lambda x: x / x.sum()).T
    fig, ax = plt.subplots(figsize=(10, 5))
    out_perc.plot(kind='bar', stacked=True, colormap=colormap, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Percentage of students')
    ax.set_xlabel(col.replace('_', ' ').title())
    ax.legend(title='Performance')
    st.pyplot(fig)
    plt.close(fig)


def plot_box(df, x, y, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_eda(df):
    st.header('Exploratory Data Analysis')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='final_grade', data=df, ax=ax)
    ax.set_title('Final grade distribution')
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = df['remarks'].value_counts().index.tolist()
    sizes = df['remarks'].value_counts().values
    colors = sns.color_palette('pastel')[0:3]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%.0f%%')
    ax.set_title('Performance category share')
    st.pyplot(fig)
    plt.close(fig)

    plot_box(df, 'remarks', 'age', 'Age distribution by performance category')

    st.subheader('Student counts by age, sex, and school')
    g = sns.catplot(x='age', hue='sex', col='school', data=df, kind='count', palette='flare', aspect=0.9)
    g.fig.suptitle('Student counts by age, sex, and school', y=1.05)
    st.pyplot(g.fig)
    plt.close(g.fig)

    plot_stacked_bar(df, 'school', 'Performance by school', 'cool')
    plot_stacked_bar(df, 'sex', 'Performance by sex', 'plasma')
    plot_stacked_bar(df, 'address', 'Performance by residence', 'YlOrBr')
    plot_stacked_bar(df, 'parents_status', 'Performance by parents cohabitation status', 'spring')
    plot_stacked_bar(df, 'mother_edu', "Performance by mother's education", 'summer')
    plot_stacked_bar(df, 'father_edu', "Performance by father's education", 'winter')
    plot_stacked_bar(df, 'mother_job', "Performance by mother's job", 'Purples')
    plot_stacked_bar(df, 'father_job', "Performance by father's job", 'viridis')
    plot_stacked_bar(df, 'travel_time', 'Performance by travel time', 'Spectral')
    plot_stacked_bar(df, 'study_time', 'Performance by study time', 'Wistia')
    plot_stacked_bar(df, 'failures', 'Performance by past failures', 'Set2')
    plot_stacked_bar(df, 'higher', 'Performance by desire for higher education', 'Set1')
    plot_stacked_bar(df, 'internet', 'Performance by internet access', 'tab10')

    plot_box(df, 'mother_job', 'final_grade', "Final grade distribution by mother's job")
    plot_box(df, 'travel_time', 'final_grade', 'Final grade distribution by travel time')

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation heatmap')
    st.pyplot(fig)
    plt.close(fig)

@st.cache_data
def build_input_df(input_values, _feature_columns):
    input_df = pd.DataFrame([input_values])
    input_df = pd.get_dummies(input_df)
    for col in _feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[_feature_columns]
    return input_df

st.title('🎓 Student Performance Predictor')
st.write('Use student demographics, study habits, and past grades to predict whether a student is likely to perform Poorly, Fairly, or Excellent.')

with st.expander('Model details'):
    st.write('This app trains a Random Forest classifier on the combined student-mat and student-por datasets, then predicts performance categories based on input features.')

df = load_data()
X, y = prepare_features(df)
model, accuracy, feature_columns = train_model(X, y)

st.metric('Model accuracy on holdout data', f'{accuracy:.2%}')

plot_eda(df)

st.sidebar.header('Student input')
student_input = {
    'school': st.sidebar.selectbox('School', ['GP', 'MS']),
    'age': st.sidebar.selectbox('Age', list(range(15, 23))),
    'address': st.sidebar.selectbox('Address', ['U', 'R']),
    'mother_edu': st.sidebar.selectbox('Mother education (Medu)', [0, 1, 2, 3, 4]),
    'father_edu': st.sidebar.selectbox('Father education (Fedu)', [0, 1, 2, 3, 4]),
    'mother_job': st.sidebar.selectbox('Mother job', ['at_home', 'health', 'other', 'services', 'teacher']),
    'father_job': st.sidebar.selectbox('Father job', ['at_home', 'health', 'other', 'services', 'teacher']),
    'reason': st.sidebar.selectbox('Reason to choose school', ['course', 'home', 'other', 'reputation']),
    'travel_time': st.sidebar.selectbox('Travel time', [1, 2, 3, 4]),
    'study_time': st.sidebar.selectbox('Study time', [1, 2, 3, 4]),
    'failures': st.sidebar.selectbox('Past class failures', [0, 1, 2, 3]),
    'higher': st.sidebar.selectbox('Wants higher education', ['yes', 'no']),
    'internet': st.sidebar.selectbox('Internet access at home', ['yes', 'no']),
    'romantic': st.sidebar.selectbox('In a romantic relationship', ['yes', 'no']),
    'go_out': st.sidebar.selectbox('Go out with friends', [1, 2, 3, 4, 5]),
    'weekday_alc': st.sidebar.selectbox('Workday alcohol consumption', [1, 2, 3, 4, 5]),
    'weekend_alc': st.sidebar.selectbox('Weekend alcohol consumption', [1, 2, 3, 4, 5]),
    'health': st.sidebar.selectbox('Current health status', [1, 2, 3, 4, 5]),
    'G1': st.sidebar.slider('First period grade (G1)', 0, 20, 10),
    'G2': st.sidebar.slider('Second period grade (G2)', 0, 20, 10)
}

input_df = build_input_df(student_input, feature_columns)
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0]
probabilities = dict(zip(model.classes_, probability))

st.subheader('Prediction')
st.write(f'**Predicted performance category:** {prediction}')
st.write('**Prediction probabilities:**')
for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    st.write(f'- {label}: {prob:.2%}')

with st.expander('Input feature values'):
    st.json(student_input)

with st.expander('Training dataset sample'):
    st.dataframe(df[['school', 'sex', 'age', 'address', 'mother_edu', 'father_edu', 'mother_job', 'father_job', 'reason', 'G1', 'G2', 'final_grade', 'remarks']].head(10))
