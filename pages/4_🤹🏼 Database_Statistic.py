import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

# 设置页面为宽屏模式
st.set_page_config(layout="wide")


# 定义不同的函数，这些函数将在选择对应选项时执行
def function1():
    st.write(" Executed> 👩‍💻**Database Version:** Bump Test: V0")

    # 加载数据并进行转置
    @st.cache_data
    def load_data():
        data = pd.read_csv('knc_bump_data_v01.csv')
        # 假设第一行是列名，第一列是车型，进行转置
        data = data.set_index(data.columns[0]).transpose()
        return data
    

    df_bump = load_data()

    # Streamlit 页面设置
    #st.title('KnC Database Analysis')

    filename = 'knc_bump_data_v01.csv'
    st.write("Loaded Filename:")
    st.text(filename)

    # 使用 multiselect 选择要分析的列
    selected_columns = st.multiselect('Select the columns for analysis:', df_bump.columns)

    # 选择分析类型
    analysis_type = st.radio(
        "Select the analysis type:",
        ('Normal Distribution Fit', 'K-means Clustering')
    )

    # 根据选择进行分析
    if analysis_type == 'Normal Distribution Fit':
        for selected_column in selected_columns:
            # 对选定的列进行正态分布拟合
            data = df_bump[selected_column].dropna()
            mu, std = norm.fit(data)
            
            # 每个列都新建一个图形
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            ax.set_title(f'Normal Distribution Fit for {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Density')
            
            # 显示图形
            st.pyplot(fig)
            
            # 关闭图形以避免重叠
            plt.close(fig)

            with st.expander("Show Parameters"):
                st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                st.write("""
                    - **Mean (µ):** The average or central value of a dataset, indicating its location.
                    - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                    - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                    - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                """)

    elif analysis_type == 'K-means Clustering':
        # 选择要创建的聚类数
        num_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3)

        if st.button('Perform K-means Clustering'):
            

            data = df_bump[selected_columns].dropna()
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
            df_bump['Cluster'] = kmeans.labels_

             # 在这里计算评估指标
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            davies_bouldin = davies_bouldin_score(data, kmeans.labels_)
            calinski_harabasz = calinski_harabasz_score(data, kmeans.labels_)
            # 展示评估指标
            st.write(f"Silhouette Coefficient: {silhouette_avg}")
            st.write(f"Davies-Bouldin Index: {davies_bouldin}")
            st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
            # 可折叠的解释部分
            with st.expander("Explanation of Cluster Evaluation Metrics"):
                st.write("""
                    - **Silhouette Coefficient:** This metric ranges from -1 to 1. A high value indicates that objects are well matched to their own cluster and poorly matched to neighboring clusters.
                    - **Davies-Bouldin Index:** A lower index signifies a better partitioning. It is the average 'similarity' between clusters, where similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
                    - **Calinski-Harabasz Index:** Also known as the Variance Ratio Criterion, a higher index relates to a model with better defined clusters.
                """)        

            # 分列显示每个聚类的统计信息
            cols = st.columns(num_clusters)
            max_num_cars = 0
            for i in range(num_clusters):
                cluster_data = df_bump[df_bump['Cluster'] == i]
                max_num_cars = max(max_num_cars, cluster_data.shape[0])
            
            for i in range(num_clusters):
                cluster_data = df_bump[df_bump['Cluster'] == i]
                with cols[i]:
                    st.write(f'Cluster {i+1}')
                    cluster_data_column = cluster_data[selected_columns]
                    # 使用 Pandas 的 describe 方法计算统计数据
                    st.dataframe(cluster_data_column.describe().transpose())

                    # 获得并显示车型数据
                    car_types = cluster_data.index.unique()
                    car_types_df = pd.DataFrame(car_types, columns=['Car Type'])
                    # 使表格的行数相等
                    car_types_df = car_types_df.reindex(np.arange(max_num_cars)).fillna('')
                    st.dataframe(car_types_df)  # 显示车型表格


                    # 绘图
                    for selected_column in selected_columns:
                        mu, std = norm.fit(cluster_data[selected_column])
                        plt.hist(cluster_data[selected_column], bins=30, density=True, alpha=0.6, color='g')
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mu, std)
                        plt.plot(x, p, 'k', linewidth=2)
                        plt.title(f'Normal Distribution for {selected_column}')
                        st.pyplot(plt)
                        plt.clf()  # 清除图形以备下次绘制

                        with st.expander("Show Parameters"):
                            st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                            st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                            st.write("""
                                - **Mean (µ):** The average or central value of a dataset, indicating its location.
                                - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                                - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                                - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                            """)
def function2():
    st.write(" Executed> 👩‍💻**Database Version:** Roll Test: V0")

    # 加载数据并进行转置
    @st.cache_data
    def load_data():
        data = pd.read_csv('knc_Roll_data_v01.csv')
        # 假设第一行是列名，第一列是车型，进行转置
        data = data.set_index(data.columns[0]).transpose()
        return data
    

    df_roll = load_data()


    # 在 Streamlit 中显示文件名
    filename = 'knc_roll_data_v01.csv'
    st.write("Loaded Filename:")
    st.text(filename)

    # 使用 multiselect 选择要分析的列
    selected_columns = st.multiselect('Select the columns for analysis:', df_roll.columns)

    # 选择分析类型
    analysis_type = st.radio(
        "Select the analysis type:",
        ('Normal Distribution Fit', 'K-means Clustering')
    )

    # 根据选择进行分析
    if analysis_type == 'Normal Distribution Fit':
        for selected_column in selected_columns:
            # 对选定的列进行正态分布拟合
            data = df_roll[selected_column].dropna()
            mu, std = norm.fit(data)
            
            # 每个列都新建一个图形
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            ax.set_title(f'Normal Distribution Fit for {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Density')
            
            # 显示图形
            st.pyplot(fig)
            
            # 关闭图形以避免重叠
            plt.close(fig)

            with st.expander("Show Parameters"):
                st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                st.write("""
                    - **Mean (µ):** The average or central value of a dataset, indicating its location.
                    - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                    - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                    - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                """)

    elif analysis_type == 'K-means Clustering':
        # 选择要创建的聚类数
        num_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3)

        if st.button('Perform K-means Clustering'):
            
            # Prepare the data by dropping NaNs and storing it for clustering
            data_for_clustering = df_roll[selected_columns].dropna()

            # Perform K-means Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data_for_clustering)

            # Create a new DataFrame for the cluster labels with the same index as the data used for clustering
            labels_df = pd.DataFrame(kmeans.labels_, index=data_for_clustering.index, columns=['Cluster'])

            # Join the cluster labels to the original DataFrame
            df_roll = df_roll.join(labels_df)

            # Compute the evaluation metrics
            silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
            davies_bouldin = davies_bouldin_score(data_for_clustering, kmeans.labels_)
            calinski_harabasz = calinski_harabasz_score(data_for_clustering, kmeans.labels_)
            # Display the evaluation metrics
            st.write(f"Silhouette Coefficient: {silhouette_avg}")
            st.write(f"Davies-Bouldin Index: {davies_bouldin}")
            st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
            # Expander for explanation of cluster evaluation metrics
            with st.expander("Explanation of Cluster Evaluation Metrics"):
                st.write("""
                    - **Silhouette Coefficient:** This metric ranges from -1 to 1. A high value indicates that objects are well matched to their own cluster and poorly matched to neighboring clusters.
                    - **Davies-Bouldin Index:** A lower index signifies a better partitioning. It is the average 'similarity' between clusters, where similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
                    - **Calinski-Harabasz Index:** Also known as the Variance Ratio Criterion, a higher index relates to a model with better defined clusters.
                """)

            # Display statistics for each cluster in separate columns
            cols = st.columns(num_clusters)
            max_num_cars = 0
            for i in range(num_clusters):
                cluster_data = df_roll[df_roll['Cluster'] == i]
                max_num_cars = max(max_num_cars, cluster_data.shape[0])

            for i in range(num_clusters):
                cluster_data = df_roll[df_roll['Cluster'] == i]
                with cols[i]:
                    st.write(f'Cluster {i+1}')
                    cluster_data_column = cluster_data[selected_columns]
                    # Compute statistics using Pandas describe method
                    st.dataframe(cluster_data_column.describe().transpose())

                    # Obtain and display car type data
                    car_types = cluster_data.index.unique()
                    car_types_df = pd.DataFrame(car_types, columns=['Car Type'])
                    # Ensure the table has an equal number of rows
                    car_types_df = car_types_df.reindex(np.arange(max_num_cars)).fillna('')
                    st.dataframe(car_types_df)  # Display car type table

                    # Plotting
                    for selected_column in selected_columns:
                        # Fit a normal distribution to the cluster data
                        cluster_data_clean = cluster_data[selected_column].dropna()  # Ensure no NaN values
                        mu, std = norm.fit(cluster_data_clean)
                        plt.hist(cluster_data_clean, bins=30, density=True, alpha=0.6, color='g')
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mu, std)
                        plt.plot(x, p, 'k', linewidth=2)
                        plt.title(f'Normal Distribution for {selected_column}')
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure for the next plot


                        with st.expander("Show Parameters"):
                            st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                            st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                            st.write("""
                                - **Mean (µ):** The average or central value of a dataset, indicating its location.
                                - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                                - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                                - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                            """)

def function3():
    st.write(" Executed> 👩‍💻**Database Version:** Lateral Force Test: V0")

    # 加载数据并进行转置
    @st.cache_data
    def load_data():
        data = pd.read_csv('knc_Lateral_data_v01.csv')
        # 假设第一行是列名，第一列是车型，进行转置
        data = data.set_index(data.columns[0]).transpose()
        return data
    

    df_lat = load_data()


    # 在 Streamlit 中显示文件名
    filename = 'knc_Lateral_data_v01.csv'
    st.write("Loaded Filename:")
    st.text(filename)

    # 使用 multiselect 选择要分析的列
    selected_columns = st.multiselect('Select the columns for analysis:', df_lat.columns)

    # 选择分析类型
    analysis_type = st.radio(
        "Select the analysis type:",
        ('Normal Distribution Fit', 'K-means Clustering')
    )

    # 根据选择进行分析
    if analysis_type == 'Normal Distribution Fit':
        for selected_column in selected_columns:
            # 对选定的列进行正态分布拟合
            data = df_lat[selected_column].dropna()
            mu, std = norm.fit(data)
            
            # 每个列都新建一个图形
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            ax.set_title(f'Normal Distribution Fit for {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Density')
            
            # 显示图形
            st.pyplot(fig)
            
            # 关闭图形以避免重叠
            plt.close(fig)

            with st.expander("Show Parameters"):
                st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                st.write("""
                    - **Mean (µ):** The average or central value of a dataset, indicating its location.
                    - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                    - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                    - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                """)

    elif analysis_type == 'K-means Clustering':
        # 选择要创建的聚类数
        num_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3)

        if st.button('Perform K-means Clustering'):
            
            # Prepare the data by dropping NaNs and storing it for clustering
            data_for_clustering = df_lat[selected_columns].dropna()

            # Perform K-means Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data_for_clustering)

            # Create a new DataFrame for the cluster labels with the same index as the data used for clustering
            labels_df = pd.DataFrame(kmeans.labels_, index=data_for_clustering.index, columns=['Cluster'])

            # Join the cluster labels to the original DataFrame
            df_lat = df_lat.join(labels_df)

            # Compute the evaluation metrics
            silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
            davies_bouldin = davies_bouldin_score(data_for_clustering, kmeans.labels_)
            calinski_harabasz = calinski_harabasz_score(data_for_clustering, kmeans.labels_)
            # Display the evaluation metrics
            st.write(f"Silhouette Coefficient: {silhouette_avg}")
            st.write(f"Davies-Bouldin Index: {davies_bouldin}")
            st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
            # Expander for explanation of cluster evaluation metrics
            with st.expander("Explanation of Cluster Evaluation Metrics"):
                st.write("""
                    - **Silhouette Coefficient:** This metric ranges from -1 to 1. A high value indicates that objects are well matched to their own cluster and poorly matched to neighboring clusters.
                    - **Davies-Bouldin Index:** A lower index signifies a better partitioning. It is the average 'similarity' between clusters, where similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
                    - **Calinski-Harabasz Index:** Also known as the Variance Ratio Criterion, a higher index relates to a model with better defined clusters.
                """)

            # Display statistics for each cluster in separate columns
            cols = st.columns(num_clusters)
            max_num_cars = 0
            for i in range(num_clusters):
                cluster_data = df_lat[df_lat['Cluster'] == i]
                max_num_cars = max(max_num_cars, cluster_data.shape[0])

            for i in range(num_clusters):
                cluster_data = df_lat[df_lat['Cluster'] == i]
                with cols[i]:
                    st.write(f'Cluster {i+1}')
                    cluster_data_column = cluster_data[selected_columns]
                    # Compute statistics using Pandas describe method
                    st.dataframe(cluster_data_column.describe().transpose())

                    # Obtain and display car type data
                    car_types = cluster_data.index.unique()
                    car_types_df = pd.DataFrame(car_types, columns=['Car Type'])
                    # Ensure the table has an equal number of rows
                    car_types_df = car_types_df.reindex(np.arange(max_num_cars)).fillna('')
                    st.dataframe(car_types_df)  # Display car type table

                    # Plotting
                    for selected_column in selected_columns:
                        # Fit a normal distribution to the cluster data
                        cluster_data_clean = cluster_data[selected_column].dropna()  # Ensure no NaN values
                        mu, std = norm.fit(cluster_data_clean)
                        plt.hist(cluster_data_clean, bins=30, density=True, alpha=0.6, color='g')
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mu, std)
                        plt.plot(x, p, 'k', linewidth=2)
                        plt.title(f'Normal Distribution for {selected_column}')
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure for the next plot


                        with st.expander("Show Parameters"):
                            st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                            st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                            st.write("""
                                - **Mean (µ):** The average or central value of a dataset, indicating its location.
                                - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                                - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                                - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                            """)    

def function4():
    st.write(" Executed> 👩‍💻**Database Version:** Longitudinal Force (Braking) Test: V0")

    # 加载数据并进行转置
    @st.cache_data
    def load_data():
        data = pd.read_csv('knc_Longitudinal_data_v01.csv')
        # 假设第一行是列名，第一列是车型，进行转置
        data = data.set_index(data.columns[0]).transpose()
        return data
    

    df_long = load_data()


    # 在 Streamlit 中显示文件名
    filename = 'knc_Lateral_data_v01.csv'
    st.write("Loaded Filename:")
    st.text(filename)

    # 使用 multiselect 选择要分析的列
    selected_columns = st.multiselect('Select the columns for analysis:', df_long.columns)

    # 选择分析类型
    analysis_type = st.radio(
        "Select the analysis type:",
        ('Normal Distribution Fit', 'K-means Clustering')
    )

    # 根据选择进行分析
    if analysis_type == 'Normal Distribution Fit':
        for selected_column in selected_columns:
            # 对选定的列进行正态分布拟合
            data = df_long[selected_column].dropna()
            mu, std = norm.fit(data)
            
            # 每个列都新建一个图形
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True, alpha=0.6, color='g')
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            ax.set_title(f'Normal Distribution Fit for {selected_column}')
            ax.set_xlabel(selected_column)
            ax.set_ylabel('Density')
            
            # 显示图形
            st.pyplot(fig)
            
            # 关闭图形以避免重叠
            plt.close(fig)

            with st.expander("Show Parameters"):
                st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                st.write("""
                    - **Mean (µ):** The average or central value of a dataset, indicating its location.
                    - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                    - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                    - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                """)

    elif analysis_type == 'K-means Clustering':
        # 选择要创建的聚类数
        num_clusters = st.slider('Select number of clusters:', min_value=2, max_value=10, value=3)

        if st.button('Perform K-means Clustering'):
            
            # Prepare the data by dropping NaNs and storing it for clustering
            data_for_clustering = df_long[selected_columns].dropna()

            # Perform K-means Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data_for_clustering)

            # Create a new DataFrame for the cluster labels with the same index as the data used for clustering
            labels_df = pd.DataFrame(kmeans.labels_, index=data_for_clustering.index, columns=['Cluster'])

            # Join the cluster labels to the original DataFrame
            df_long = df_lat.join(labels_df)

            # Compute the evaluation metrics
            silhouette_avg = silhouette_score(data_for_clustering, kmeans.labels_)
            davies_bouldin = davies_bouldin_score(data_for_clustering, kmeans.labels_)
            calinski_harabasz = calinski_harabasz_score(data_for_clustering, kmeans.labels_)
            # Display the evaluation metrics
            st.write(f"Silhouette Coefficient: {silhouette_avg}")
            st.write(f"Davies-Bouldin Index: {davies_bouldin}")
            st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
            # Expander for explanation of cluster evaluation metrics
            with st.expander("Explanation of Cluster Evaluation Metrics"):
                st.write("""
                    - **Silhouette Coefficient:** This metric ranges from -1 to 1. A high value indicates that objects are well matched to their own cluster and poorly matched to neighboring clusters.
                    - **Davies-Bouldin Index:** A lower index signifies a better partitioning. It is the average 'similarity' between clusters, where similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
                    - **Calinski-Harabasz Index:** Also known as the Variance Ratio Criterion, a higher index relates to a model with better defined clusters.
                """)

            # Display statistics for each cluster in separate columns
            cols = st.columns(num_clusters)
            max_num_cars = 0
            for i in range(num_clusters):
                cluster_data = df_long[df_long['Cluster'] == i]
                max_num_cars = max(max_num_cars, cluster_data.shape[0])

            for i in range(num_clusters):
                cluster_data = df_long[df_long['Cluster'] == i]
                with cols[i]:
                    st.write(f'Cluster {i+1}')
                    cluster_data_column = cluster_data[selected_columns]
                    # Compute statistics using Pandas describe method
                    st.dataframe(cluster_data_column.describe().transpose())

                    # Obtain and display car type data
                    car_types = cluster_data.index.unique()
                    car_types_df = pd.DataFrame(car_types, columns=['Car Type'])
                    # Ensure the table has an equal number of rows
                    car_types_df = car_types_df.reindex(np.arange(max_num_cars)).fillna('')
                    st.dataframe(car_types_df)  # Display car type table

                    # Plotting
                    for selected_column in selected_columns:
                        # Fit a normal distribution to the cluster data
                        cluster_data_clean = cluster_data[selected_column].dropna()  # Ensure no NaN values
                        mu, std = norm.fit(cluster_data_clean)
                        plt.hist(cluster_data_clean, bins=30, density=True, alpha=0.6, color='g')
                        xmin, xmax = plt.xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x, mu, std)
                        plt.plot(x, p, 'k', linewidth=2)
                        plt.title(f'Normal Distribution for {selected_column}')
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure for the next plot


                        with st.expander("Show Parameters"):
                            st.write(f"**Mean (µ):** The average or central value of the dataset for `{selected_column}` is {mu:.2f}.")
                            st.write(f"**Standard Deviation (σ):** Measures the amount of variation or dispersion of the dataset for `{selected_column}` is {std:.2f}.")
                            st.write("""
                                - **Mean (µ):** The average or central value of a dataset, indicating its location.
                                - **Standard Deviation (σ):** The measure of variability or dispersion in a dataset, indicating spread.
                                - **Histogram:** A chart that displays the distribution of data, with each bar representing the frequency of data within a specific interval.
                                - **Density Plot:** This curve is an estimate of the data distribution, with its shape determined by the mean and standard deviation.
                            """)        

def function5():
    st.write(" Executed> 👩‍💻**Database Version:** Align Torque Test: V0")

# 模拟加载过程的函数
def run_loading_animation():
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)  # 等待时间可以调整
        progress_bar.progress(percent_complete + 1)
    progress_bar.empty()  # 加载完成后清除进度条

# 创建网页
st.title('KnC Database Analysis')

# 显示一行文字
st.write('Welcome to the KnC Database Analysis page.')

# 显示可以折叠的内容
with st.expander("See explanation"):
    st.write("""
         This is some explanation of the analysis that can be hidden or shown.
         """)

# 创建一个Selectbox控件
option = st.selectbox(
    'Choose a function to execute',
    ('Choose a Test', 'Bump Test', 'Roll Test', 'Lateral Force Test', 'Longitudinal Force (Braking) Test', 'Align Torque Test')  # 初始为空选项
)

# 当用户选择一个选项后，显示加载动画，然后执行对应的函数
if option:
    # 显示读取动画，到100后再执行函数
    run_loading_animation()

    # 根据选择执行不同的函数
    if option == 'Bump Test':
        function1()
    elif option == 'Roll Test':
        function2()
    elif option == 'Lateral Force Test':
        function3()
    elif option == 'Longitudinal Force (Braking) Test':
        function4()
    elif option == 'Align Torque Test':
        function5()
