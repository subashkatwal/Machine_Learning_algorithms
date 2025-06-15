import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load and convert
image = cv2.imread("tiktok.webp")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape((-1, 3))

# Apply KMeans
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixels)

# Get labels and centers
labels = kmeans.labels_
centers = np.uint8(kmeans.cluster_centers_)

# Reshape labels to image shape
labels_image = labels.reshape(image.shape[:2])

# Display each cluster as a separate image
plt.figure(figsize=(12, 6))
for i in range(k):
    mask = labels_image == i
    clustered_img = np.zeros_like(image)
    clustered_img[mask] = centers[i]
    plt.subplot(1, k, i + 1)
    plt.imshow(clustered_img)
    plt.title(f'Cluster {i+1}')
    plt.axis("off")
plt.tight_layout()
plt.show()
