#include "error_code.h"
#include <cstdarg>

namespace EC {
	ErrorCode::ErrorCode() :
		status(0),
		message()
	{ }

	ErrorCode::ErrorCode(const char* format, ...) : status{ -1 } {
		va_list argptr;
		va_start(argptr, format);
		makeFormat(format, argptr);
		va_end(argptr);
	}

	ErrorCode::ErrorCode(int status, const char* format, ...) : status{ status } {
		va_list argptr;
		va_start(argptr, format);
		makeFormat(format, argptr);
		va_end(argptr);
	}

	void ErrorCode::makeFormat(const char* format, va_list argptr) {
		const int defaultSize = 1024;
		char formated[defaultSize] = { '\0' };
		const int size = vsprintf(formated, format, argptr);
		if (size > defaultSize) {
			message.resize(size);
			vsprintf(&message[0], format, argptr);
		} else {
			message = std::string(formated);
		}
	}
}
