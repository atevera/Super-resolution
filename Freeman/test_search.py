import example_based_super_resolution as sr
import time

f = sr.h5py.File('training_data.h5', 'r')

patchesHR = f.get('patches')
vectorsID = f.get('vectors')

vector_search = vectorsID[7]
vector_search[0] -= 1e-6

start = time.time()

print('Realizando búsqueda. Espere por favor ... ')


nbrs = sr.NearestNeighbors(n_neighbors = 1, algorithm = 'ball_tree').fit(vectorsID)

distances, indices = nbrs.kneighbors([vector_search])

print(distances, indices)
end = time.time()
print('Tiempo de búsqueda requerido {}'.format(end - start))