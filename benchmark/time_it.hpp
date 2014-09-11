#ifndef LINALG_TIME_IT_HPP_
#define LINALG_TIME_IT_HPP_

#include <iostream>
#include <chrono>
#include <ratio>

template <typename F>
void time_it(F f){
	namespace chrono = std::chrono;
	auto t0 = chrono::system_clock::now();
	f();
	auto t1 = chrono::system_clock::now();
	auto diff = chrono::duration_cast<chrono::microseconds>(t1 - t0);
	std::cout << "1 run, taking "
				<< diff.count() << " microseconds." << std::endl;
}

template <typename F>
void time_it(F f, const int n){

	int total = 0;
	for (int i = 0; i < n; ++i){
		namespace chrono = std::chrono;
		auto t0 = chrono::system_clock::now();
		f();
		auto t1 = chrono::system_clock::now();
		auto diff = chrono::duration_cast<chrono::milliseconds>(t1 - t0);
		total += diff.count();
	}
	std::cout << n << " runs, taking an average of "
				<<  static_cast<double>(total)/(1000 * n) << " secs." << std::endl;
}

#endif
