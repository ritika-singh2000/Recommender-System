# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

def main():
    st.title("Welcome to TheSocialComment Recomendation System 😃 ")
    st.markdown("The system will recommed you, once you give your name  or  the name of the title of the post you liked !")
    # load the data
    @st.cache(persist=True)
    def load_data(my_data):
        df = pd.read_csv("{}.csv".format(my_data))
        return df


    users=load_data("users")
    posts = load_data("posts")
    #rename the columns
    users.rename(columns={'_id':'user_id'} , inplace = True)

    # Enter  the users name
    name_list = users.name.to_list()
    name_list.insert(0,"")
    # select box to select the name of the user
    user_name = st.selectbox(
    'Enter name of User :',
    (name_list)
     )
    # Enter the title name
    title_list = posts.title.to_list()
    title_list.insert(0,"")
    # select box to enter the post name
    title_name = st.selectbox(
    'Enter the Title of the Post :',
    (title_list)
     )

     # select the number of the recommendation the user needs
    number_of_rec = st.number_input("Enter the number of recommedation of posts you would like to have :", 1, 15, step=1, key='n_estimators')
    number_of_rec= number_of_rec+1
    # loading the pivot table which was saved from the notebook "users_pivot_table_01"
    users_pivot_table = pd.read_csv('users_pivot_table_01')
    if user_name !='':
        query_index = users_pivot_table.index[users_pivot_table['name'] == user_name]
        print(query_index[0])
    # setting the 'name' column as the index
    users_pivot_table.set_index('name', inplace=True)

    # For the posts recomendation
    final_posts=pd.read_csv('final_posts_01', sep='\t')
    tfv = TfidfVectorizer(min_df = 3 , max_features  = None , strip_accents = 'unicode' , analyzer = 'word',
                      ngram_range=(1,3) , stop_words = 'english')
    tfv_matrix = tfv.fit_transform(final_posts['data'])
    sig = sigmoid_kernel(tfv_matrix , tfv_matrix)
    indices = pd.Series(final_posts.index , index = final_posts['title'] ).drop_duplicates()
    # function to provide the recommendation
    def recommendation(title , sig=sig):
                  idx = indices[title]
                  sig_scores = list(enumerate(sig[idx]))
                  sig_scores = sorted(sig_scores , key = lambda x : x[1] , reverse =  True)
                  sig_scores = sig_scores[1:number_of_rec]
                  movies_indies = [i[0] for i in sig_scores]
                  x = (final_posts['title'].iloc[movies_indies])
                  x.to_list()
                  for i in range(len(x)):
                      st.write('{0}: {1}'.format(i+1 ,x.values[i]))

    if st.checkbox("Recommend for the Post!", False):
       # The recommendation for the post
       if title_name == '':
           st.write("Sorry :( ....You need to choose one of the post.")
       else:
               st.markdown("Recommendation for the post : **{}**".format(title_name))
               recommendation(title_name)


    # providing a button which helps to recommend
    if st.checkbox("Recommend for the User !",False):
        # The recommendation for the post
        # st.markdown("Recommendation for the post : **{}**".format(title_name))
        # recommendation(title_name)
        # The recommendation for the users
        if user_name == '':
             st.write("Sorry :( ....You need to choose your name")
        else:
            users_final_df = csr_matrix(users_pivot_table.values)
            model_knn = NearestNeighbors(metric = 'cosine' , algorithm = 'brute')
            model_knn.fit(users_final_df)
            distances , indices  = model_knn.kneighbors(users_pivot_table.iloc[query_index,:].values.reshape(1,-1) , n_neighbors=number_of_rec)

            for i in range(0, len(distances.flatten())):
                    if i == 0:
                        st.markdown('Recommendations for **{0}**:\n'.format(users_pivot_table.index[query_index][0]))
                    else:
                        st.write('{0}: {1}'.format(i, users_pivot_table.columns[indices.flatten()[i]]))

if __name__ == '__main__':
    main()
