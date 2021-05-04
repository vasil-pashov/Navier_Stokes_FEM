#define _USE_MATH_DEFINES
#include "expression.h"
#include "error_code.h"
#include <cctype>
#include <cmath>

namespace NSFem {

struct Token {
	enum class Type {
		Invalid,
		Number,
		Variable,
		Plus,
		Minus,
		Multiply,
		Divide,
		Power,
		Sin,
		Cos,
		OpenParen,
		CloseParen
	};
	bool isOperator() const {
		switch (t) {
			case Type::Plus:
			case Type::Minus:
			case Type::Multiply:
			case Type::Divide:
			case Type::Power:
			case Type::Sin:
			case Type::Cos:
				return true;
			default: return false;
		}
	}
	Token(const float val) : val(val), t(Type::Number) {}
	Token(const char name) : name(name), t(Type::Variable) {}
	Token(const Type t) : t(t) {}
	union {
		float val;
		char name;
	};
	Type t;
};

inline static Expression::Operator tokenToOperator(const std::vector<Token>& tokens, const int tokenIndex) {
	const Token::Type t = tokens[tokenIndex].t;
	switch (t) {
		case Token::Type::Plus : return Expression::Operator::Plus;
		case Token::Type::Power : return Expression::Operator::Power;
		case Token::Type::Divide : return Expression::Operator::Divide;
		case Token::Type::Multiply : return Expression::Operator::Multiply;
		case Token::Type::Sin : return Expression::Operator::Sin;
		case Token::Type::Cos : return Expression::Operator::Cos;
		case Token::Type::Minus: {
			const bool isFirst = tokenIndex == 0;
			if (isFirst ||
				(!isFirst && tokens[tokenIndex - 1].t == Token::Type::OpenParen) ||
				(!isFirst && tokens[tokenIndex - 1].isOperator())
			) {
				return Expression::Operator::UnaryMinus;
			} else {
				return Expression::Operator::Minus;
			}
		} break;
		default: {
			assert(false);
			return Expression::Operator::Invalid;
		}
	}
}

/// Evaluate binary operator. In infix notation this would read left op right. Assuming the operator is left associative
/// @param[in] op The Expression::Operator code for the operator
/// @param[in] left The left operand for the operator
/// @param[in] right The right operand for the operator
/// @retval nan if wrong op is passed (i.e. for unary operator) otherwise the result of left op right
inline static float evaluateOperator(const Expression::Operator op, const float left, const float right) {
	switch (op) {
		case Expression::Operator::Plus: return left + right;
		case Expression::Operator::Minus: return left - right;
		case Expression::Operator::Multiply: return left * right;
		case Expression::Operator::Divide: return left / right;
		case Expression::Operator::Power: return powf(left, right);
		default: {
			assert(false);
			return nanf("");
		}
	}
}

/// Evaluate unary operator
/// @param[in] op The Expression::Operator code for the operator
/// @param[in] operand The single operand for the operator
/// @retval nan if the op is wrong (i.e. its for binary operator) otherwise the result of applying the operator to the operand
inline static float evaluateOperator(const Expression::Operator op, const float operand) {
	switch (op) {
		case Expression::Operator::Sin: return sinf(operand);
		case Expression::Operator::Cos: return cosf(operand);
		case Expression::Operator::UnaryMinus: return -operand;
		default: {
			assert(false);
			return nanf("");
		}
	}
}

/// Given operator return the number of operand it needs
/// @retval -1 if wrong op is passed, 1 for unary operators, 2 for binary operators
inline static int8_t getNumOperatorArgs(const Expression::Operator op) {
	switch (op) {
		case Expression::Operator::Plus:
		case Expression::Operator::Minus:
		case Expression::Operator::Multiply:
		case Expression::Operator::Divide:
		case Expression::Operator::Power:
			return 2;
		case Expression::Operator::Sin:
		case Expression::Operator::Cos:
		case Expression::Operator::UnaryMinus:
			return 1;
		default: {
			assert(false);
			return -1;
		}
	}
}

/// Check if the first symbols of start contain some of the allowed operators
/// @param[in] start The string where we look for operators
/// @param[out] len The length of the operator string if no operator is found it will be 0
/// @retval Expression::Operator::Invalid if no operator is found otherwise Expression::Operator value for the operator
inline static Token::Type parseOperator(const char* start, int& len) {
	const char c = start[0];
	if (c == '+') {
		len = 1;
		return Token::Type::Plus;
	} else if (c == '-') {
		len = 1;
		return Token::Type::Minus;
	} else if (c == '/') {
		len = 1;
		return Token::Type::Divide;
	} else if (c == '*') {
		len = 1;
		return Token::Type::Multiply;
	} else if (c == '^') {
		len = 1;
		return Token::Type::Power;
	} else if (c == '(') {
		len = 1;
		return Token::Type::OpenParen;
	} else if (c == ')') {
		len = 1;
		return Token::Type::CloseParen;
	} else if (!strncmp(start, "sin", 3)) {
		len = 3;
		return Token::Type::Sin;
	} else if (!strncmp(start, "cos", 3)) {
		len = 3;
		return Token::Type::Cos;
	}
	len = 0;
	return Token::Type::Invalid;
}

/// Check if the first symbols of start contain some of the allowed variable names. Currently they are x y z t
/// @param[in] start The string where we look for variables
/// @param[out] len The length of the variable. If no variable is found this will be 0
/// @retval 1 if variable name is found 0 otherwise
inline static bool parseVariable(const char* start, int& len) {
	if (*start == 'x' || *start == 'y' || *start == 'z' || *start == 't') {
		len = 1;
		return true;
	}
	len = 0;
	return false;
}

/// Check if the fist symbols of start conain some symbolc constant and return it
/// @param[in] start The string where we look for constants
/// @param[out] len The length of the constant if no constant is found it will be 0
/// @retval 0 if no constant is found, otherwise the value of the found constant
inline static float parseConstant(const char* start, int& len) {
	if (start[0] == 'E') {
		len = 1;
		return M_E;
	} else if (start[0] == 'P' && start[1] == 'i') {
		len = 2;
		return M_PI;
	}
	len = 0;
	return 0;
}

inline static int getOperatorPrecedence(Expression::Operator op) {
	switch (op) {
		case Expression::Operator::Plus:
		case Expression::Operator::Minus:
			return 0;
		case Expression::Operator::Multiply:
		case Expression::Operator::Divide:
			return 1;
		case Expression::Operator::UnaryMinus:
			return 2;
		case Expression::Operator::Power:
			return 3;
		case Expression::Operator::Sin:
		case Expression::Operator::Cos:
			return 10;
		default: {
			return -1;
			assert(false);
		}
	}
}

/// Tells if we have to pop operators before pushing the given operator on the stack
/// @param[in] stack The stack where the operator is going to be pushed
/// @param[in] current The operator for which we check if poping is needed
/// @retval 1 if we need to pop operators before pushing current to the stack, 0 if we don't need to pop
inline static bool shouldPop(const std::vector<Expression::Operator>& stack, const Expression::Operator current) {
	if (stack.empty()) return false;
	const int topPrecedence = getOperatorPrecedence(stack.back());
	const int currentPrecendece = getOperatorPrecedence(current);
	if (current == Expression::Operator::UnaryMinus) return false;
	// If the last seen operator has higher precedence it must consume its operands, then it
	// must be added to the tree as a node. That node will be an operand of the current operator
	// This way we preserve precedence.
	if (topPrecedence >= currentPrecendece) return true;
	return false;
}

static EC::ErrorCode tokenize(const char* expression, std::vector<Token>& tokens) {
	const char* it = expression; // Iterator for the expression. Holds the next char to be parsed
	Token::Type type; // Used to keep the result from parseOperator
	int length = 0; // Used to keep the length of parsed tokens in order to advance the iterator after parsing
	float value; // Used to keep values for symbolic and numeric conastants
	char* end; // Used to keep the end for std::strof
	while (*it) {
		if ((type = parseOperator(it, length)) != Token::Type::Invalid) {
			tokens.emplace_back(type);
			it += length;
		} else if ((value = parseConstant(it, length))) {
			tokens.emplace_back(value);
			it += length;
		} else if ((value = std::strtof(it, &end)) || end != it) {
			tokens.emplace_back(value);
			it = end;
		} else if (parseVariable(it, length)) {
			tokens.emplace_back(it[0]);
			it++;
		} else {
			return EC::ErrorCode(
				"Error while parsing expression: %s. Unexpected symbol: %c at %d",
				expression,
				it[0],
				it - expression
			);
		}
		while (std::isspace(it[0])) it++;
	}
	return EC::ErrorCode();
}

EC::ErrorCode Expression::popOperators(
	std::vector<Expression::Operator>& operatorStack,
	std::vector<int>& pendingOperands,
	std::vector<Node>& tree
) {
	const Expression::Operator op = operatorStack.back();
	const int numArguments = getNumOperatorArgs(op);
	if (numArguments == 1) {
		if (pendingOperands.empty()) {
			return EC::ErrorCode("Error while parsing expression. Expected 1 operand got 0");
		}
		// This is unary operator the operand will always be at the previous index in the tree
		// Substitute the last operand in the tree with the result of the operator. Since that
		// result will be added at the end of the tree array we use the size to index the new
		// node and the push the new node into the tree.
		pendingOperands.back() = tree.size();
		tree.emplace_back(op);
	} else if (numArguments == 2) {
		if (pendingOperands.size() < 2) {
			return EC::ErrorCode("Error while parsing expression. Expected 2 operands. Got only one.");
		}
		// Consume the two operands and form a new node in the tree. Then substitute the result
		// push the result of the operator in the list with pending operands. The right-hand side
		// of the new node will always be at index one less than the the index of new node, while
		// the left hand side is somewhare to the left of the node (and the right-hand side)

		// Consume the right hand side
		pendingOperands.pop_back();

		// Consume the left hand side
		const int left = pendingOperands.back();
		pendingOperands.pop_back();

		// Create new node, push it at the end of the tree array and push the result of that new
		// node to the list of pending operands.
		pendingOperands.push_back(tree.size());
		tree.emplace_back(left, op);
	}
	operatorStack.pop_back();
	return EC::ErrorCode();
}

EC::ErrorCode Expression::init(const char* expression) {
	// This is a stack to hold the operands which are to be combined in an expression
	// An operand is any node in the expression tree. If the node is a leaf then the
	// value of the operand is the value held in that leaf, if the node not a leaf then
	// the value of the operand is the value from the subtree with root the given node.
	// The operands are stored by their element index (position) in the tree. The top
	// of this stack holds the first operand which must be included in the new node of
	// the expression.
	std::vector<int> pendingOperands;
	// Stack which holds the operators which must be applied to the operands in pendingOperands
	// The top of this stack is the next operator which must be applied.
	std::vector<Operator> operatorStack;
	// This will hold tokenized version of the string expression. The tokens are in the
	// same order as in the string version of the expression.
	std::vector<Token> tokens;
	RETURN_ON_ERROR_CODE(tokenize(expression, tokens));
	// The loop which creates the tree. The tree is build bottom up all values and variables are leaves.
	// At each step check the current operand. If it's a value/variable push it to the tree as a leaf and
	// push that node into the pendingOperands array. If it's operator there are several options:
	// (i) The operator stack has operator with higher precendece. In this case that operator must be
	// squashed into a node i.e. it must consume its operands from pendingOperands and this must be added
	// as a node in the tree. This operation must be executed while all operators on the stack have higher
	// precedence than the current operator. This operation forces all operators with higher precedence to
	// be evaluated before operators with lower precedence. Then go to (ii)
	// (ii) There are no operators on the stack with higher precedence than the current operator. In this
	// case we add the operator on the operator stack and procede.
	// (iii) The operator is open parentheses. In this case just add it to the stack
	// (iv) The operator is close parentheses. We will form an expression tree with all operators and operands
	// in the parentheses. That tree is going be added as e regular intermediate node in the expression tree.
	for (int i = 0; i < tokens.size(); ++i) {
		const Token& t = tokens[i];
		if (t.isOperator()) {
			const Expression::Operator op = tokenToOperator(tokens, i);
			while (shouldPop(operatorStack, op)) {
				RETURN_ON_ERROR_CODE(popOperators(operatorStack, pendingOperands, tree));
			}
			operatorStack.push_back(op);
			// We assume that function arguments are surrounded by () this would ensure that we will first evaluate
			// what is in the () since they override precedence and then the result of what is in the () will be passed
			// to the unary function
			if (op == Expression::Operator::Sin || op == Expression::Operator::Cos) {
				if (i >= tokens.size() || tokens[i + 1].t != Token::Type::OpenParen) {
					return EC::ErrorCode(
						"Error parsing expression: %s. Functions must be surrounded by ()",
						expression
					);
				}
			}
		} else if (t.t == Token::Type::Number) {
			pendingOperands.push_back(tree.size());
			tree.emplace_back(t.val);
		} else if (t.t == Token::Type::Variable) {
			pendingOperands.push_back(tree.size());
			tree.emplace_back(t.name);
		} else if (t.t == Token::Type::OpenParen) {
			operatorStack.emplace_back(Expression::Operator::OpenParen);
		} else if (t.t == Token::Type::CloseParen) {
			while (operatorStack.size() && operatorStack.back() != Operator::OpenParen) {
				RETURN_ON_ERROR_CODE(popOperators(operatorStack, pendingOperands, tree));
			}
			if (operatorStack.empty()) {
				return EC::ErrorCode("Error parsing expression: %s. Mismatched brackets.", expression);
			}
			operatorStack.pop_back();
		}
	}
	// The loop above does not squash all operators and operands into a tree. It just creates nodes and
	// handles operator precedence. After the loop is done we might be left with some unsquashed operators
	// for example if all operators have the same precendence. This final loop iterates over all operators
	// and operands and produces the final tree.
	while (operatorStack.size()) {
		popOperators(operatorStack, pendingOperands, tree);
	}
	// At the end of the algorithm pendingOperands must have one operand which is the root of the tree
	if (pendingOperands.size() != 1) {
		return EC::ErrorCode("Wrong expression: %s", expression);
	}
	return EC::ErrorCode();
}

EC::ErrorCode Expression::evaluate(const std::unordered_map<char, float>* variables, float& outResult) const {
	return evaluate(variables, outResult, tree.size() - 1);
}

EC::ErrorCode Expression::evaluate(
	const std::unordered_map<char, float>* variables,
	float& outResult,
	const int nodeIdx
) const {
	const Node& node = tree[nodeIdx];
	if (node.isLeaf()) {
		if (node.isSymbolic()) {
			if (!variables) {
				return EC::ErrorCode("Missing variable: %c. No variables table is passed at all", node.getName());
			}
			const std::unordered_map<char, float>::const_iterator it = variables->find(node.getName());
			if (it == variables->end()) {
				return EC::ErrorCode("Missing variable: %c", node.getName());
			} else {
				outResult = it->second;
				return EC::ErrorCode();
			}
		} else {
			outResult = node.getValue();
			return EC::ErrorCode();
		}
	}
	const uint8_t numOperands = node.getNumberOfArguments();
	if (numOperands == 1) {
		float childRes = 0;
		RETURN_ON_ERROR_CODE(evaluate(variables, childRes, nodeIdx - 1));
		outResult = evaluateOperator(node.getOperator(), childRes);
	} else if (numOperands == 2) {
		float left = 0, right = 0;
		RETURN_ON_ERROR_CODE(evaluate(variables, left, node.getLeftIndex()));
		RETURN_ON_ERROR_CODE(evaluate(variables, right, nodeIdx - 1));
		outResult = evaluateOperator(node.getOperator(), left, right);
	} else {
		assert(false);
	}
	return EC::ErrorCode();
}

}