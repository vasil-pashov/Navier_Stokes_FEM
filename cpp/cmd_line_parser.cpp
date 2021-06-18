#include "cmd_line_parser.h"
#include "error_code.h"
#include <cstring>
#include <cassert>

namespace CMD {

CommandLineArgs::ParamInfo::ParamInfo(
	const char* description,
	const Type type,
	const bool required
) :
	description(std::string(description)),
	type(type),
	required(required)
{ }

EC::ErrorCode CommandLineArgs::parse(const int argc, char** argv) {
    // argv[0] is the name of the program (exe), that is why the for starts from 1
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg[0] == '-') {
            int j = 1;
            bool hasParams = false;
            while (arg[j]) {
                if (arg[j] == '=') {
                    hasParams = true;
                    break;
                }
                j++;
            }

            // -someArg=someVal -> arg + 1 to start past the '-' from 's'
            // j-1 is the length, substract 1 to account for the parameter
            const int nameLen = j - 1;
            std::string name(arg + 1, nameLen);
            InfoConstIt it = paramInfo.find(name);
            if (it == paramInfo.end()) {
                return EC::ErrorCode(-1, "Unknown parameter %s", name.c_str());
            }
            if (!hasParams && it->second.type == Type::None) {
                paramValues[name] = ParamVal();
                continue;
            } else if (hasParams && it->second.type == Type::None) {
                return EC::ErrorCode(
                    -1,
                    "Parameter is a flag parameter. It does not have value!"
                    "Received input is: %s",
                    arg
                );
            } else if (!hasParams && it->second.type != Type::None) {
                return EC::ErrorCode(
                    -1,
                    "Parameter must receive value. The syntax is -parameter=value (no spaces surrounding =)"
                    "Received input is: %s",
                    arg
                );
            }
            
            ParamVal val;
            const int paramStartIndex = j + 1;
            switch (it->second.type) {
                case Type::Int: {
                    char* next;
                    const int res = strtol(arg + paramStartIndex, &next, 0);
                    // strtol retuns 0 if it cannot parse, so there are two possibilities to get 0
                    // a) It could not parse
                    // b) It parsed a zero
                    // So chech if the number is 0
                    // second param of strtol returns which is the next char after the last parsed char
                    // we want it to be the end of the string
                    if ((res == 0 && arg[j + 1] != '0') || *next != '\0') {
                        return EC::ErrorCode(
                            -1,
                            "Wrong parameter value. Expected: -param=[INTEGER_VALUE]"
                            "Received input is: %s",
                            arg
                        );
                    }
                    val.intVal = res;
                } break;
                case Type::String: {
                    // Don't forget it starts with '-'
                    const int len = strlen(arg) - j;
                    val.stringVal = new char[len + 1];
                    strcpy(val.stringVal, arg + paramStartIndex);
                    val.stringVal[len] = '\0';
                } break;
                default: {
                    assert(false);
                    return EC::ErrorCode(
                        -1,
                        "Unknown input %s. All arguments must start with \'-\' and use \'=\' for setting value."
                        "There must be no intervals surrounding \'=\' or after the \'-\'!",
                        arg
                    );
                }
            }
            paramValues[name] = val;
        } else {
            return EC::ErrorCode(-1,
                "Unknown input %s. All arguments must start with \'-\' and use \'=\' for setting value."
                "There must be no intervals surrounding \'=\' or after the \'-\'!",
                arg
            );
        }
    }
    for (auto& it : paramInfo) {
        if (it.second.required && paramValues.find(it.first) == paramValues.end()) {
            return EC::ErrorCode(
                -1,
                "Required parameter %s cannot be found!",
                it.first.c_str()
            );
        }
    }
    return EC::ErrorCode();
}

CommandLineArgs::~CommandLineArgs() {
	freeMem();
}

void CommandLineArgs::addParam(
	const char* name,
	const char* description,
	const Type type,
	const bool required
) {
	paramInfo[std::string(name)] = ParamInfo(description, type, required);
}

const int* CommandLineArgs::getIntVal(const char* name) const {
	const std::string paramName(name);
	ValuesConstIt it = paramValues.find(paramName);
	if (it != paramValues.end()) {
		return &it->second.intVal;
	}
	return nullptr;
}

const char* CommandLineArgs::getStringVal(const char* name) const {
	const std::string paramName(name);
	ValuesConstIt it = paramValues.find(paramName);
	if (it != paramValues.end()) {
		return it->second.stringVal;
	}
	return nullptr;
}

bool CommandLineArgs::isSet(const char* name) const {
	return paramValues.find(std::string(name)) != paramValues.end();
}

void CommandLineArgs::freeValues() {
	for (auto& it : paramValues) {
		const bool isString = paramInfo[it.first].type == Type::String;
		if (isString) {
			delete[] it.second.stringVal;
		}
	}
	paramValues.clear();
}

void CommandLineArgs::freeMem() {
	freeValues();
	paramInfo.clear();
}

void CommandLineArgs::print() {
	for (auto& it : paramInfo) {
		printf("%s - %s\n", it.first.c_str(), it.second.description.c_str());
	}
}

} // namespace CMD