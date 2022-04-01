import example_based_super_resolution as sr
import time
import matplotlib.pyplot as plt


print('Iniciando proceso de súper resolución. Por favor espere...')
alpha = 0.1*((7^2)/(2*5-1))
start = time.time()
alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print('Se utilizará un alpha de {}'.format(alpha))

for n in range(1,14):
    print('Procesando imagen {}...'.format(n))
    image = 'test_LR\im' + str(n) + '.jpg'
    img_sr = sr.superresolution(image, 4, alpha)
    sr.cv2.imwrite('test_LR/results/im'+str(n)+'.jpg',img_sr.superresolution)
    sr.cv2.imwrite('test_LR/results/im'+str(n)+'HF.jpg', img_sr.high_frequencies)

end = time.time()
print('Tiempo requerido {}'.format(end - start))