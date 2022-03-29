from sklearn.decomposition import KernelPCA

def calculate_kpca_of_gradients(logger, gradients, num_components):
    kpca = KernelPCA(n_components=num_components,kernel='cosine')

    logger.info("Computing {}-component KPCA of gradients".format(num_components))

    return kpca.fit_transform(gradients)
