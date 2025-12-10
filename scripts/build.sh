#!/bin/bash

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/../build"
FAISS_DIR="${ROOT}/../faiss"
FAISS_BUILD="${FAISS_DIR}/build"

RESET=$1

if [[ "$RESET" == "r" ]]; then
	rm -rf ${BUILD} ${FAISS_BUILD}
	echo "cleared ${BUILD} and ${FAISS_BUILD}"
elif [[ "$RESET" == "w" ]]; then
	echo "SWIG Only BUILD"
	cd $FAISS_BUILD/faiss/python;
	python setup.py install;
	exit
fi

if [[ -d "${BUILD}" ]]; then
    echo "build folder already exists. Skipping CMake configuration."
else
    echo "build folder does not exist. Running CMake configuration..."
	cd ${FAISS_DIR}
	cmake -B ${FAISS_BUILD} \
		-DCMAKE_BUILD_TYPE=Debug \
		-DFAISS_ENABLE_GPU=ON \
		-DFAISS_ENABLE_CUVS=OFF \
		-DFAISS_ENABLE_PYTHON=ON \
		-DFAISS_ENABLE_CACHEFLOW=ON \
		-DFAISS_OPT_LEVEL=avx2 \
		-DBLA_VENDOR=Intel10_64_dyn;
fi

cd ${FAISS_DIR}
make -C ${FAISS_BUILD} -j faiss_avx2 faiss_gpu swigfaiss_avx2 -s;
cd ${FAISS_BUILD}/faiss/python;
python setup.py install;
cd ${ROOT}/..