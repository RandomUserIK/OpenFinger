cd ./fingervein-preprocessing/ && rm -rf build && mkdir build && cd build && cmake .. && make && ./unittests && cd ../..
cd ./fingervein-extraction/ && rm -rf build && mkdir build && cd build && cmake .. && make && ./unittests && cd ../..
cd ./fingervein-matching/ && rm -rf build && mkdir build && cd build && cmake .. && make && ./unittests