#pragma once
#include <unordered_map>
#include <string>

namespace EC {
    class ErrorCode;
}

namespace CMD {

/// Structure to parse and hold command line arguments
struct CommandLineArgs {
	~CommandLineArgs();
	enum class Type {
		Int,
		String,
		None
	};
	/// parse command line arguments that come from main
	/// @param[in] numArgs the number of the arguments
	/// @param[in] args  array of arguments that comes from main function where
	///                  args[0] is the name of the program, args[1] is the first argument and so on
	EC::ErrorCode parse(const int numArgs, char** args);
	/// Add a info for possible parameter that can come from the command line. All possible parameters must
	/// be added before calling parse. Calling this does not add a parameter only information.
	/// @param[in] name name of the param, must not include - or =.
	/// @param[in] description description for the parameter can include everything. This is shown when calling print
	/// @param[in] type type of the parameter. Int types are extracted with getIntVal, string tipes with getStringVal ...
	/// @param[in] required if true the parameter is required. If a parameter is required but not found parse will return error
	void addParam(const char* name, const char* description, const Type type, const bool required);
	/// Try to extract parameter.
	/// @param[in] name the name of the parameter
	/// @retval pointer to the value or nullptr if the param is not present
	const int* getIntVal(const char* name) const;
	/// Try to extract parameter.
	/// @param[in] name the name of the parameter
	/// @retval pointer to the value or nullptr if the param is not present
	const char* getStringVal(const char* name) const;
	/// Check if the parameter has been parsed. This is the way to check if flag parameter (param with no type) has been set.
	/// @param[in] name the name of the parameter
	/// @retval true if the parameter was parsed, false otherwise
	bool isSet(const char* name) const;
	/// Show the parameters with their description
	/// @param[in,out] f File stream where parameters will be printed
	void print(FILE* f);
	/// Delete all parameters as well as all parameter info
	void freeMem();
	/// Delete only the parameters, without deleting the info
	void freeValues();
private:
	struct ParamInfo {
		ParamInfo() = default;
		ParamInfo(const char* description, const Type type, const bool required);
		std::string description;
		Type type;
		bool required;
	};
	union ParamVal {
		int intVal;
		char* stringVal;
	};
	std::unordered_map<std::string, ParamInfo> paramInfo;
	std::unordered_map<std::string, ParamVal> paramValues;
	using ValuesConstIt = std::unordered_map<std::string, ParamVal>::const_iterator;
	using InfoConstIt = std::unordered_map<std::string, ParamInfo>::const_iterator;
};

}
