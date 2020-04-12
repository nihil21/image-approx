import cv2
import numpy as np
from numpy.linalg import svd, LinAlgError
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def approximate_matrix(mtx, k, name='current'):
    """Given a matrix and the number k of singular values to keep, it performs the SVD
    and returns the approximated matrix"""
    print('Performing SVD for {0:s} matrix...'.format(name))
    [u, s, vt] = svd(mtx, full_matrices=False)

    print('{0:d} eigenvalues found'.format(len(s)))

    a_k = np.zeros(shape=(u.shape[0], vt.shape[1]))
    for i in range(0, k):
        a_k += s[i] * np.outer(u[:, i], vt[i, :])
    return a_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help='relative path to the image')
    ap.add_argument('-d', '--degree', required=False,
                    help='degree of approximation (optional, by default it is equal to 2)')
    args = vars(ap.parse_args())

    # Argument check
    path = args['path']
    k = args['degree']
    if k is None:
        print('Using default value (2) for k...')
        k = 2
    else:
        try:
            k = int(args['degree'])
            if k < 0:
                sys.exit('The degree must be a positive integer.')
        except ValueError:
            sys.exit('The degree must be a positive integer.')

    # Read image
    in_img = cv2.imread(path)
    if in_img is None:
        sys.exit('Image not found')
    # Convert to double
    in_img = cv2.normalize(in_img.astype(np.float), None, 0., 1., cv2.NORM_MINMAX)
    # Extract BGR matrices
    mtxB = in_img[:, :, 0]
    mtxG = in_img[:, :, 1]
    mtxR = in_img[:, :, 2]

    # Concurrently perform SVD for each channel
    print('Performing SVD on each channel, it may require several minutes depending on the resolution...')
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futureB = executor.submit(approximate_matrix, mtxB, k, "blue")
            futureG = executor.submit(approximate_matrix, mtxG, k, "green")
            futureR = executor.submit(approximate_matrix, mtxR, k, "red")
            futures = [futureB, futureG, futureR]
            for future in as_completed(futures):
                if future == futureB:
                    mtxB_k = future.result()
                elif future == futureG:
                    mtxG_k = future.result()
                else:
                    mtxR_k = future.result()
        print('SVD approximation computed.')
    except LinAlgError:
        sys.exit('SVD could not converge, try with a different image.')

    # Merge channels to produce new image
    out_img = np.zeros(shape=in_img.shape, dtype=np.float)
    out_img[:, :, 0] = mtxB_k
    out_img[:, :, 1] = mtxG_k
    out_img[:, :, 2] = mtxR_k
    out_img = cv2.normalize(out_img, None, 0, 255, cv2.NORM_MINMAX)
    # Save image to file
    cv2.imwrite('out.jpg', out_img)
    print('Output image saved to out.jpg.')


if __name__ == '__main__':
    main()
