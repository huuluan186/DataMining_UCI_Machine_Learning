import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import json
import os

def save_state(selected_num, selected_cat):
    with open("state.json", "w") as f:
        json.dump({
            "selected_num_cols": selected_num,
            "selected_cat_cols": selected_cat
        }, f)

def load_state():
    if os.path.exists("state.json"):
        with open("state.json", "r") as f:
            state = json.load(f)
        return state.get("selected_num_cols", []), state.get("selected_cat_cols", [])
    return numerical_cols, categorical_cols  # fallback mặc định


# ----------------------
# CẤU HÌNH GIAO DIỆN
# ----------------------
st.set_page_config(page_title="Phân cụm học sinh", layout="wide")
st.title("🎓 Phân cụm sinh viên dựa trên các đặc điểm học tập của họ, từ đó khám phá các nhóm sinh viên có hiệu suất tương tự")

# ----------------------
# SIDEBAR – CHỌN k CỤM
# ----------------------
with st.sidebar:
    st.header("🔧 Cài đặt")
    chosen_k = st.slider("Chọn số cụm k", min_value=2, max_value=10, value=4)

# ----------------------
# TẢI & LÀM SẠCH DỮ LIỆU
# ----------------------
@st.cache_data

def load_data():
    df1= pd.read_csv("./data/student-mat.csv")
    df2 = pd.read_csv("./data/student-por.csv", sep=";")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

df = load_data()
original_len = len(df)

# --- loại G3 <= 0 và outlier ---
df = df[df["G3"] > 0]
filtered_len = len(df)

# Tùy chọn cột số
num_cols = ["absences","G3","G1","G2"]

# Loại bỏ outlier trên toàn bộ các cột số (dùng IQR)
df_cleaned = df.copy()
for col in num_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]

cleaned_len = len(df_cleaned)


# ----------------------
# TIỀN XỬ LÝ & PHÂN CỤM
# ----------------------
categorical_cols = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic",
]
numerical_cols = [
    "Medu", "Fedu", "studytime", "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "traveltime" ,"G3"
]


# 1. Chia thành numerical và categorical data
X_num = df_cleaned[numerical_cols]
X_cat = df_cleaned[categorical_cols]

# 2. Chuẩn hóa numerical data
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 3. One-hot encode categorical data
encoder = OneHotEncoder(sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

# 4. Ghép lại thành ma trận cuối cùng
X_scaled = np.hstack([X_num_scaled, X_cat_encoded])

# --- tính PCA & phân cụm ---
#Tạo một mô hình PCA với số chiều muốn giữ lại là x thành phần chính
pca_model = PCA(n_components=2)
pca_2d = pca_model.fit_transform(X_scaled)
pca_model_3d = PCA(n_components=3)
pca_3d = pca_model_3d.fit_transform(X_scaled)

# ==== KMEANS ====
kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)
df_cleaned["Cluster"] = clusters

# ----------------------
# GÁN NHÃN HỌC LỰC (chỉ để HẬU KIỂM)
# ----------------------
def level(g):
    if g < 6:   return "Yếu"
    elif g < 11: return "Trung bình"
    elif g < 16: return "Khá"
    else:        return "Giỏi"

df_cleaned["G3_level"] = df_cleaned["G3"].apply(level)

# Danh sách đặc trưng
numerical_features = numerical_cols  # 12 cột
categorical_features = list(encoder.get_feature_names_out(categorical_cols))  # 26 cột
all_features = numerical_features + categorical_features
 # Khởi tạo session_state
if "selected_num_cols" not in st.session_state:
    st.session_state.selected_num_cols = numerical_cols
if "selected_cat_cols" not in st.session_state:
    st.session_state.selected_cat_cols = categorical_cols
if "ok_clicked" not in st.session_state:
    st.session_state.ok_clicked = False
if "X_scaled_selected" not in st.session_state:
    st.session_state.X_scaled_selected = X_scaled
if "selected_features" not in st.session_state:
    st.session_state.selected_features = all_features
if "clusters_selected" not in st.session_state:
    st.session_state.clusters_selected = clusters

# Hiển thị bảng G3_level vs Cluster ở đầu mỗi tab
if st.session_state.ok_clicked:
    df_cleaned["Cluster"] = st.session_state.clusters_selected

ct = pd.crosstab(df_cleaned["Cluster"], df_cleaned["G3_level"], normalize="index") * 100
g3_mean = df_cleaned.groupby("Cluster")["G3"].mean().round(2)
ct["Trung bình G3"] = g3_mean


# Copy bảng gốc để hiển thị
ct_display = ct.copy()

# Cột cần rename (hiển thị)
ct_display = ct.rename(columns={
    "Giỏi": "Giỏi (16-20)",
    "Khá": "Khá (11-15)",
    "Trung bình": "Trung bình (6-10)",
    "Yếu": "Yếu (0-5)"
})

# Các cột cần highlight
cols_pct_renamed = ["Giỏi (16-20)", "Khá (11-15)", "Trung bình (6-10)", "Yếu (0-5)"]

# Subheader
st.subheader("🔗 G3_level vs Cluster")

# Hiển thị với định dạng % nhưng dữ liệu vẫn là số float → highlight đúng
st.write(
    ct_display.style
    .format({
        "Trung bình G3": "{:.2f}",
        "Giỏi (16-20)": "{:.1f}%",
        "Khá (11-15)": "{:.1f}%",
        "Trung bình (6-10)": "{:.1f}%",
        "Yếu (0-5)": "{:.1f}%"
    })
    .highlight_max(subset=cols_pct_renamed, axis=1, color="lightgreen")
)

# Tính purity vẫn như cũ
purity = ct[["Giỏi", "Khá", "Trung bình", "Yếu"]].max(axis=1).mean() / 100
st.markdown(f"✨ **Purity trung bình**: {purity:.1%}")

# ----------------------
# GIAO DIỆN BÊN TRÁI – CHỌN BƯỚC
# ----------------------
steps = [
    "1️⃣ Dữ liệu ban đầu và làm sạch",
    "2️⃣ Phân tích Elbow & Silhouette",
    "3️⃣ Phân cụm và PCA Visualization",
    "4️⃣ Số lượng học sinh mỗi cụm",
    "5️⃣ 📘 Biến định tính (Categorical)",
    "6️⃣ 📙 Biến định lượng (Numerical)",
    "7️⃣ 🔍 Top N đặc trưng gốc",
    "8️⃣ 🔍 Khám phá đặc trưng mỗi cụm ",
    #"🔍 Kiểm tra"
]

with st.sidebar:
    st.subheader("📋 Bước phân tích")
    chosen_step = st.radio("Chọn bước muốn xem:", steps)

# ----------------------
# THANH TIẾN TRÌNH
# ----------------------
st.markdown(f"## 📊 {chosen_step}")
progress = (steps.index(chosen_step) + 1) / len(steps)
st.progress(progress)

# ----------------------
# BƯỚC 1: Dữ liệu ban đầu và làm sạch
# ----------------------
if chosen_step == steps[0]:
    # ⚠️ Kiểm tra giá trị thiếu 
    st.markdown("### 🧼 Kiểm tra giá trị thiếu:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Không có giá trị thiếu.")

    # 👁️ Hiển thị thông tin thống kê
    st.markdown(f"**Số lượng bản ghi ban đầu:** {original_len}")
    st.markdown(f"**Số lượng bản ghi sau khi loại bỏ G3 ≤ 0:** {filtered_len}")
    st.markdown(f"**Số lượng bản ghi trước khi loại bỏ outliers:** {filtered_len}")
    st.markdown(f"**Số lượng sau khi loại outlier tất cả biến số:** {cleaned_len}")
    st.markdown(f"**🧹 Đã loại bỏ:** {filtered_len - cleaned_len} bản ghi có ngoại lai.")

    # Đánh dấu outlier
    df["is_outlier"] = ~df.index.isin(df_cleaned.index)

    # Trong phần giao diện BƯỚC 1, thêm checkbox
    if chosen_step == steps[0]:
        if st.checkbox("📋 Hiển thị dữ liệu outlier"):
            st.dataframe(df[df["is_outlier"]])
    st.markdown(f"**Số lượng bản ghi sau khi loại bỏ outliers:** {cleaned_len}")

    # ✅ Hiển thị dữ liệu nếu người dùng muốn
    if st.checkbox("📋 Hiển thị dữ liệu sau khi làm sạch"):
        st.dataframe(df_cleaned.head())
    if chosen_step == steps[0]:
        if st.checkbox("📋 Miêu tả dữ liệu"):
            st.dataframe(df_cleaned.iloc[:, :-2].describe())
    # 🎯 Phân bố G3 trước xử lý
    fig_g3_before, ax = plt.subplots(figsize=(5, 2))
    df_original = load_data()  # Lấy dữ liệu ban đầu
    g3_counts = df_original["G3"].value_counts().sort_index()
    ax.scatter(g3_counts.index, g3_counts.values, alpha=0.7, color="blue", s=25)
    ax.set_xlim(-0.5, 20.5)
    ax.set_xticks(np.arange(0, 21, 1)) 
    ax.set_yticks(np.arange(0, g3_counts.values.max() + 10, 20))
    ax.set_xlabel("G3 (Điểm cuối kỳ)")
    ax.set_ylabel("Số lượng")
    ax.set_title("G3 (trước tiền xử lý)")
    ax.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig_g3_before)

    # ✅ Phân bố G3 SAU khi làm sạch  (không dùng X_processed)
    fig_g3_after, ax2 = plt.subplots(figsize=(5, 2))
    g3_after_counts = df_cleaned["G3"].value_counts().sort_index()
    ax2.scatter(g3_after_counts.index, g3_after_counts.values,
                alpha=0.7, color="seagreen", s=25)

    ax2.set_xlim(-0.5, 20.5)
    ax2.set_xticks(np.arange(0, 21, 1)) 
    ax2.set_yticks(np.arange(0, g3_after_counts.values.max() + 10, 20))
    ax2.set_xlabel("G3 (đã lọc outlier)")
    ax2.set_ylabel("Số lượng")
    ax2.set_title("G3 (sau làm sạch)")
    ax2.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig_g3_after)

# ----------------------
# BƯỚC 2: Phân tích tương quan, chọn biến, Elbow & Silhouette
# ----------------------
elif chosen_step == steps[1]:
    st.markdown("### 📉 Phân tích tương quan với G3")
    # Kiểm tra shape
    st.write(f"Shape X_scaled: {X_scaled.shape}")
    st.write(f"Tổng số đặc trưng: {len(all_features)}")
    if len(all_features) != X_scaled.shape[1]:
        st.error(f"Lỗi: all_features có {len(all_features)} cột, X_scaled có {X_scaled.shape[1]} cột.")
        st.stop()

    # DataFrame cho tương quan
    X_all = pd.DataFrame(X_scaled, columns=all_features)
    X_all["G3_original"] = df_cleaned["G3"].values

    # 1. Tương quan biến định lượng
    st.markdown("#### 📊 Tương quan biến định lượng")
    corr_num = df_cleaned[numerical_cols].corr(method="pearson")["G3"].sort_values(ascending=False)
    corr_num_df = pd.DataFrame({"Biến": corr_num.index, "Tương quan": corr_num.values})
    st.dataframe(corr_num_df.style.format({"Tương quan": "{:.3f}"}))

    # Heatmap biến định lượng
    fig_num_heatmap, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_cleaned[numerical_cols].corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Tương quan biến định lượng")
    st.pyplot(fig_num_heatmap)

    # 2. Tương quan biến định tính
    st.markdown("#### 📊 Tương quan biến định tính")
    cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=categorical_features)
    cat_encoded_df["G3"] = df_cleaned["G3"].values
    corr_cat = cat_encoded_df.corr(method="pearson")["G3"].drop("G3").sort_values(ascending=False)
    corr_cat_df = pd.DataFrame({"Biến": corr_cat.index, "Tương quan": corr_cat.values})
    st.dataframe(corr_cat_df.style.format({"Tương quan": "{:.3f}"}))

    # Heatmap biến định tính (top 10)
    top_cat_features = corr_cat.index[:10]
    fig_cat_heatmap, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cat_encoded_df[list(top_cat_features) + ["G3"]].corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Tương quan biến định tính (Top 10)")
    st.pyplot(fig_cat_heatmap)

    # 3. Chọn biến cho KMeans
    st.markdown("### 🔧 Chọn biến để phân cụm")

    # Widget chọn biến
    selected_num_cols = st.multiselect(
        "Chọn biến định lượng:",
        numerical_cols,
        default=st.session_state.selected_num_cols,
        key="num_cols_select"
    )
    selected_cat_cols = st.multiselect(
        "Chọn biến định tính:",
        categorical_cols,
        default=st.session_state.selected_cat_cols,
        key="cat_cols_select"
    )

    # Nút OK
    if st.button("OK"):
        if not selected_num_cols and not selected_cat_cols:
            st.warning("Chọn ít nhất một biến!")
            st.session_state.ok_clicked = False
        else:
            # Cập nhật session_state
            st.session_state.selected_num_cols = selected_num_cols
            st.session_state.selected_cat_cols = selected_cat_cols
            st.session_state.ok_clicked = True

            # Tiền xử lý dữ liệu đã chọn
            X_num_selected = df_cleaned[selected_num_cols] if selected_num_cols else np.array([]).reshape(len(df_cleaned), 0)
            scaler = StandardScaler()
            X_num_scaled_selected = scaler.fit_transform(X_num_selected) if selected_num_cols else X_num_selected

            X_cat_selected = df_cleaned[selected_cat_cols] if selected_cat_cols else np.array([]).reshape(len(df_cleaned), 0)
            encoder_selected = OneHotEncoder(sparse_output=False)
            X_cat_encoded_selected = encoder_selected.fit_transform(X_cat_selected) if selected_cat_cols else X_cat_selected
            selected_cat_features = list(encoder_selected.get_feature_names_out(selected_cat_cols)) if selected_cat_cols else []

            # Ghép dữ liệu
            st.session_state.X_scaled_selected = np.hstack([X_num_scaled_selected, X_cat_encoded_selected])
            st.session_state.selected_features = selected_num_cols + selected_cat_features

            # Cập nhật KMeans với biến đã chọn
            kmeans_selected = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
            st.session_state.clusters_selected = kmeans_selected.fit_predict(st.session_state.X_scaled_selected)

    # Hiển thị thông tin
    st.write(f"Shape X_scaled_selected: {st.session_state.X_scaled_selected.shape}")
    st.write(f"Đặc trưng đã chọn: {st.session_state.selected_features}")

    # 4. Elbow & Silhouette
    if st.session_state.ok_clicked:
        st.markdown("### 📈 Elbow & Silhouette")
        k_range = range(2, 7)
        sse, sil = [], []
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=20).fit(st.session_state.X_scaled_selected)
            sse.append(km_tmp.inertia_)
            sil.append(silhouette_score(st.session_state.X_scaled_selected, km_tmp.labels_))

        sil_df = pd.DataFrame({"k": k_range, "Silhouette": sil}).set_index("k")
        st.dataframe(sil_df.style.highlight_max(axis=0, color="lightgreen"))

        fig_elbow, ax1 = plt.subplots(figsize=(3, 3))
        ax1.plot(k_range, sse, marker="o")
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("k")
        ax1.set_ylabel("SSE")
        st.pyplot(fig_elbow)

        fig_sil, ax2 = plt.subplots(figsize=(3, 3))
        ax2.plot(k_range, sil, marker="o", color="orange")
        ax2.set_title("Silhouette Scores")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Silhouette")
        st.pyplot(fig_sil)

        st.markdown(f"**✅ k tốt nhất: {sil_df['Silhouette'].idxmax()}**")
    else:
        st.info("Nhấn 'OK' để tính Elbow & Silhouette.")

# ----------------------
# BƯỚC 3: Phân cụm và PCA Visualization
# ----------------------
elif chosen_step == steps[2]:
    st.subheader("3️⃣ Phân cụm và PCA Visualization")

    kmeans_selected = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    if st.session_state.ok_clicked:
        # Kiểm tra và cập nhật df_cleaned["Cluster"] với cụm từ biến đã chọn
        if len(st.session_state.clusters_selected) != len(df_cleaned):
            st.warning("Số lượng cụm không khớp với dữ liệu. Kiểm tra lại biến đã chọn!")
        else:
            X_for_clustering = st.session_state.X_scaled_selected
            st.session_state.clusters_selected = kmeans_selected.fit_predict(X_for_clustering)
            df_cleaned["Cluster"] = st.session_state.clusters_selected.astype(int)  # Ép về int để loại NaN

        # Tính lại PCA 2D từ dữ liệu đã chọn
        pca_model_2d = PCA(n_components=2)
        pca_2d = pca_model_2d.fit_transform(st.session_state.X_scaled_selected)

        # Tính lại PCA 3D từ dữ liệu đã chọn
        pca_model_3d = PCA(n_components=3)
        pca_3d = pca_model_3d.fit_transform(st.session_state.X_scaled_selected)

        # Tính centroid từ cluster_centers_ và áp dụng PCA
        centroids_2d = pca_model_2d.transform(kmeans_selected.cluster_centers_)
        centroids_3d = pca_model_3d.transform(kmeans_selected.cluster_centers_)
    else:
        # Dùng dữ liệu ban đầu nếu chưa nhấn OK
        df_cleaned["Cluster"] = clusters.astype(int)
        centroids_2d = np.vstack([pca_2d[clusters == i].mean(axis=0) for i in range(chosen_k)])
        centroids_3d = np.vstack([pca_3d[clusters == i].mean(axis=0) for i in range(chosen_k)])

    # -------------------- PCA 2D bằng matplotlib --------------------
    fig_pca2d, ax = plt.subplots(figsize=(6, 3.5))
    sns.scatterplot(
        x=pca_2d[:, 0],
        y=pca_2d[:, 1],
        hue=df_cleaned["Cluster"].astype(str),  
        palette="Set2",
        ax=ax,
        s=40,
    )
    ax.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        s=50,
        c="black",
        marker="X",
        label="Centroid",
    )

    # Add labels for each centroid
    for i, (x, y) in enumerate(centroids_2d):
        ax.text(
            x + 0.05,  # Slight offset in x for readability
            y + 0.05,  # Slight offset in y for readability
            f"C{i}",  # Label as "Cụm 1", "Cụm 2", etc.
            fontsize=10,
            color="black",
            ha="left",
            va="bottom",
        )

    ax.set_title("PCA 2D với tâm cụm")
    ax.legend(title="Cụm", fontsize="small", labelspacing=0.5)
    plt.tight_layout()
    st.pyplot(fig_pca2d)

    # -------------------- PCA 3D --------------------
    st.markdown("### 🧊 PCA 3D – quan sát tổng thể")

    df_pca = pd.DataFrame(pca_3d, columns=["PC1", "PC2", "PC3"])
    df_pca["Cluster"] = df_cleaned["Cluster"].astype(str)
    df_pca["idx"] = df_pca.index
    df_pca["G3"] = df_cleaned["G3"].values

    # Vẽ PCA 3D bằng plotly
    color_sequence = px.colors.qualitative.Set2
    fig3d = px.scatter_3d(
        df_pca, x="PC1", y="PC2", z="PC3",
        color="Cluster",
        color_discrete_sequence=color_sequence,
        opacity=0.85,
        size_max=10,
        title="✨ PCA 3D",
        hover_data=["G3"]
    )

    # Thêm centroid vào biểu đồ
    centroid_trace = go.Scatter3d(
        x=centroids_3d[:, 0],
        y=centroids_3d[:, 1],
        z=centroids_3d[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='black', symbol='x'),
        text=[f'C{i}' for i in range(chosen_k)],
        textposition='top center',
        name='Centroids'
    )
    fig3d.add_trace(centroid_trace)

    # Cập nhật layout
    fig3d.update_layout(
        title="✨ PCA 3D Interactive with Centroids",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode="cube"),
        legend=dict(title="Cụm")
    )

    st.plotly_chart(fig3d, use_container_width=True)
# ----------------------
# BƯỚC 4: Số lượng học sinh mỗi cụm
# ----------------------
elif chosen_step == steps[3]:
    st.markdown("### 👥 Số lượng học sinh trong từng cụm")

    if st.session_state.ok_clicked:
        count_df = pd.Series(st.session_state.clusters_selected).value_counts().reset_index()
        count_df.columns = ["Tên cụm", "Số học sinh"]
        st.dataframe(count_df)

        fig_count, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=count_df, x="Tên cụm", y="Số học sinh", palette="Set2", ax=ax)
        ax.set_title("Số học sinh theo cụm")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=7)
        st.pyplot(fig_count)
    else:
        st.warning("Vui lòng nhấn 'OK' ở Bước 2 để chọn biến và phân cụm!")


# ----------------------
# BƯỚC 5: Biến định tính (Categorical)
# ----------------------
elif chosen_step == steps[4]:
    if st.session_state.ok_clicked:
        # Chọn một biến để vẽ biểu đồ, chỉ từ biến đã chọn
        selected_cat = st.selectbox("🔎 Chọn biến định tính để trực quan hóa:", st.session_state.selected_cat_cols)
        # Hiển thị bảng tỷ lệ phần trăm từng giá trị theo cụm
        st.markdown(f"**📌 {selected_cat}:**")
        tab = pd.crosstab(st.session_state.clusters_selected, df_cleaned[selected_cat], normalize="index") * 100
        tab.index.name = "Cluster"
        st.dataframe(tab.style.format("{:.1f}%").highlight_max(axis=1, color="lightgreen"))

        st.markdown("### 📈 Biểu đồ phân phối theo cụm")
        df_plot = df_cleaned.copy()
        df_plot["Cluster"] = st.session_state.clusters_selected
        fig_cat, ax = plt.subplots(figsize=(5, 3.5))
        chart = sns.countplot(data=df_plot, x=selected_cat, hue="Cluster", palette="Set3", ax=ax)

        # Hiển thị số lượng trên đầu mỗi cột
        for p in chart.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8)

        ax.set_title(f"Phân phối '{selected_cat}' theo cụm")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig_cat)
    else:
        st.warning("Vui lòng nhấn 'OK' ở Bước 2 để chọn biến và phân cụm!")

# ----------------------
# BƯỚC 6: Biến định lượng (Numerical)
# ----------------------
elif chosen_step == steps[5]:
    st.subheader("6️⃣ 📙 Biến định lượng (Numerical)")

    if st.session_state.ok_clicked:
        # Hiển thị bảng trung bình các biến định lượng theo cụm
        df_cleaned["Cluster"] = st.session_state.clusters_selected
        numerical_summary = df_cleaned.groupby('Cluster')[st.session_state.selected_num_cols].mean().round(2)
        st.markdown("**📊 Trung bình các biến định lượng theo cụm:**")
        st.dataframe(numerical_summary.style.highlight_max(axis=0, color='lightblue'))

        # Chọn một biến cụ thể để vẽ biểu đồ boxplot, chỉ từ biến đã chọn
        selected_num = st.selectbox("🔎 Chọn biến định lượng để xem phân phối", st.session_state.selected_num_cols)

        fig_num, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df_cleaned, x="Cluster", y=selected_num, palette="pastel", ax=ax)
        ax.set_title(f"Phân phối {selected_num} theo cụm")
        st.pyplot(fig_num)
    else:
        st.warning("Vui lòng nhấn 'OK' ở Bước 2 để chọn biến và phân cụm!")

# ----------------------
# BƯỚC 7: Top N đặc trưng gốc phân biệt các cụm
# ----------------------
elif chosen_step == steps[6]:
    st.subheader("7️⃣ 🔍 Top N đặc trưng gốc phân biệt các cụm")

    if st.session_state.ok_clicked:
        # ---------- 1) Random‑Forest importance ----------
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)
        rf.fit(st.session_state.X_scaled_selected,
               st.session_state.clusters_selected)

        feat_df = (pd.DataFrame({
                      "Feature": st.session_state.selected_features,
                      "Importance": rf.feature_importances_})
                   .sort_values(by="Importance", ascending=False))

        top_n = st.slider("Chọn số đặc trưng", 1, len(feat_df), 5)
        st.dataframe(feat_df.head(top_n)
                     .style.highlight_max(subset=["Importance"],
                                          color="lightgreen"))

        fig_feat, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=feat_df.head(top_n), x="Importance", y="Feature",
                    palette="Blues_r", ax=ax)
        ax.set_title("Top đặc trưng quan trọng nhất")
        st.pyplot(fig_feat)

        # ---------- 2) Thống kê mô tả định lượng ----------
        df_cleaned["Cluster"] = st.session_state.clusters_selected
        num_summary = (df_cleaned
                       .groupby("Cluster")[st.session_state.selected_num_cols]
                       .agg(["mean", "median", "std", "min", "max"])
                       .round(2))
        st.markdown("**📊 Thống kê mô tả các biến định lượng theo cụm:**")
        st.dataframe(num_summary.style.highlight_max(axis=0,
                                                     color="lightblue"))

        fig_heat, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(num_summary.xs("mean", level=1, axis=1),
                    annot=True, fmt=".2f", cmap="coolwarm",
                    cbar_kws={"label": "Giá trị trung bình"}, ax=ax)
        ax.set_title("Trung bình các biến định lượng theo cụm")
        st.pyplot(fig_heat)

        # ---------- 3) Bảng % từng biến định tính ----------
        st.markdown("**📊 Tỷ lệ phần trăm từng biến định tính theo cụm:**")

        # Danh sách biến định tính gốc, không trùng lặp
        cat_cols = list(dict.fromkeys(st.session_state.selected_cat_cols))

        for col in cat_cols:
            if col in categorical_cols:            # bảo đảm là biến gốc
                tab = (pd.crosstab(df_cleaned["Cluster"],
                                   df_cleaned[col],
                                   normalize="index") * 100)
                tab.index.name = "Cluster"
                st.markdown(f"**🔹 {col}:**")
                st.dataframe(tab.style
                                .format("{:.1f}%")
                                .highlight_max(axis=1, color="lightgreen"))
    else:
        st.warning("Vui lòng nhấn 'OK' ở Bước 2 để chọn biến và phân cụm!")


# ----------------------
# BƯỚC 8: Khám phá nhận xét đặc trưng cụm
# ----------------------
elif chosen_step == steps[7]:
    st.subheader("8️⃣ 🔍 Khám phá nhận xét đặc trưng cụm")

    if st.session_state.ok_clicked:
        df_cleaned["Cluster"] = st.session_state.clusters_selected
        g3_mean = df_cleaned.groupby("Cluster")["G3"].mean().round(2)
        numerical_summary = df_cleaned.groupby("Cluster")[st.session_state.selected_num_cols].mean().round(2)

        # Lấy danh sách biến định tính gốc từ selected_features
        selected_cat_cols = list(set(col.split('_')[0] for col in st.session_state.selected_features if '_' in col and col.split('_')[0] in categorical_cols))

        @st.cache_data
        def compute_cat_summary(df_cleaned, selected_cat_cols):
            cat_summary = {}
            for col in selected_cat_cols:
                if col in categorical_cols:
                    crosstab = pd.crosstab(df_cleaned["Cluster"], df_cleaned[col], normalize="index") * 100
                    cat_summary[col] = crosstab
            return cat_summary

        if "cat_summary" not in st.session_state:
            st.session_state.cat_summary = compute_cat_summary(df_cleaned, selected_cat_cols)
        cat_summary = st.session_state.cat_summary

        # Hiển thị thông tin cho từng cụm
        for cluster in range(chosen_k):
            st.markdown(f"### Cụm {cluster}")
            st.markdown(f"**Trung bình G3**: {g3_mean[cluster]:.2f}")

            # Biến định lượng nổi bật
            top_num = numerical_summary.loc[cluster].sort_values(ascending=False)
            st.markdown("**📊 Biến định lượng nổi bật**")
            st.table(top_num.reset_index().rename(columns={"index": "Biến", cluster: "Giá trị"}))

            # Biến định tính nổi bật cho cụm hiện tại
            st.markdown(f"**Biến định tính nổi bật**:")
            cat_results = []
            for col in selected_cat_cols:
                if col in cat_summary and not cat_summary[col].empty:
                    top_cat = cat_summary[col].loc[cluster].idxmax()
                    max_pct = cat_summary[col].loc[cluster].max()
                    cat_results.append({"Biến": col, "Giá trị nổi bật": top_cat, "Tỷ lệ (%)": f"{max_pct:.1f}%"})
            if cat_results:
                cat_df = pd.DataFrame(cat_results)
                st.table(cat_df)
            else:
                st.write(f"Không có biến định tính nào được tìm thấy cho cụm {cluster}.")

    else:
        st.warning("Vui lòng nhấn 'OK' ở Bước 2 để chọn biến và phân cụm!")