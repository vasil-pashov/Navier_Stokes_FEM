#pragma once
#include <string>

// Simple struct use to wrap errors.
// Contains error code for the error and message that describes it
// If status is NOT zero assume that there is error

#define RETURN_ON_ERROR_CODE(Err) \
	do { \
		EC::ErrorCode _Error = Err; \
		if(_Error.hasError()) { \
			return _Error; \
		} \
	} while(false);

namespace EC {
	class ErrorCode {
	public:
		ErrorCode();
		ErrorCode(const char*, ...);
		ErrorCode(int status, const char*, ...);
		int getStatus() const;
		const char* getMessage() const;
		bool hasError() const;
		operator bool() const {
			return hasError();
		}
	private:
		void makeFormat(const char* format, va_list argptr);
		int status;
		std::string message;
	};

	inline int ErrorCode::getStatus() const {
		return status;
	}

	inline const char* ErrorCode::getMessage() const {
		return message.c_str();
	}

	inline bool ErrorCode::hasError() const {
		return status != 0;
	}
}
