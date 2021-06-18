#pragma once
#include <vector>
#include <cstring>
#include <cassert>
#include <unordered_map>

namespace EC {
	class ErrorCode;
}

namespace NSFem {
	class Expression {
	public:
		enum class Operator : unsigned char {
			Invalid = 0,
			Plus,
			Minus,
			UnaryMinus,
			Multiply,
			Divide,
			Power,
			Sin,
			Cos,
			OpenParen,
			CloseParen
		};
	public:
		Expression() = default;
		Expression(Expression&&) = default;
		Expression& operator=(Expression&&) = default;
		Expression(const Expression&) = delete;
		Expression& operator=(const Expression&) = delete;
		EC::ErrorCode init(const char* expression);
		/// Evaluate the expression.
		/// @param[in] variables Map which holds key-value pairs between all variables in the expression
		/// and values which the user provides for them. It can be null if the expression does not have
		/// any variables in it
		/// @param[out] outResult The result of the expression
		/// @returns ErrorCode for the operation
		EC::ErrorCode evaluate(const std::unordered_map<char, float>* variables, float& outResult) const;
	private:
		class Node {
		public:
			/// Create leaf with numeric value
			/// @param[in] v the numeric value for the leaf
			explicit Node(float v);
			/// Create symbolic leaf
			/// @param[in] name The name of the variable
			explicit Node(char name);
			/// Create intermediate node for unary function
			/// @param[in] op Operator code for the function.
			explicit Node(Expression::Operator op);
			/// Create intermediate node for binary operator
			/// @param[in] leftIndex Index in the array of nodes where the left child is. The right child is always the one before this node
			/// @param[in] op Operator code for the binary operator
			Node(int leftIndex, Expression::Operator op);
			/// Check if node is a leaf
			/// @retval 1 - if it is a leaf 0 - otherwise
			bool isLeaf() const;
			/// For intermediate node return the number of arguments
			/// @retval 1 for unary operators 2 for binary operators
			uint8_t getNumberOfArguments() const;
			/// For a leaf check if it is symbolic
			/// @retval 1 if the leaf is symbolic variable 0 if it numeric variable
			bool isSymbolic() const;
			/// For an intermediate leaf get the operator
			/// @retval Operator code for the operator
			Expression::Operator getOperator() const;
			/// For an intermediate node get the index of the left operand
			/// @retval index of the left operand in the tree which is represented as an array
			int getLeftIndex() const;
			/// For a numeric leaf get the value in the leaf. One must check beforehand if it is numeric leaf.
			/// @retval the value in the leaf as float
			float getValue() const;
			/// For a symbolic leaf get the name of the variable. One must check beforehand if it is symbolic leaf.
			/// @retval The name of the symbolic variable in the leaf
			char getName() const;
		private:
			enum Flags {
				Leaf = 0,
				Intermediate = 1,
				TwoOperands = 1 << 1,
				SymbolicValue = 1 << 2
			};
			union {
				float value;
				/// For operators with two operands keep explicitly only the left operand
				/// The right operand will always be the previous node in the array
				int leftIndex;
				char variableName;
			};
			/// The first bit of flags tells us if the node is leaf - 0 or intermediate - 1
			/// The second bit tells us if the operator is unary - 0 or binary - 1
			/// The third bit tells us if we have numeric value - 0 or symbolic - 1
			/// The remaining 5 bits are left for the operator
			unsigned char flags;
		};
		/// Given operator stack and the pending operands. Pop the top most operator and create new node in the tree
		EC::ErrorCode popOperators(
			std::vector<Expression::Operator>& operatorStack,
			std::vector<int>& pendingOperands,
			std::vector<Node>& tree
		);
		EC::ErrorCode evaluate(const std::unordered_map<char, float>* variables, float& outResult, const int index) const;
		/// The tree of the expression is linearized into this array. The tree is represented in a "backwards" fashion.
		/// The root of the expression tree is the last element in this array. Leaf nodes in the tree represent either
		/// values or "variables" which must be substituted with values provided by the user during the evaluation.
		/// Intermediate nodes represent unary or binary expression. If the expression is unary expression the operant
		/// will always be at the previous index in the array. If the expression is binary the right hand side operand
		/// will always be at the previous index in the array, while the left hand side operand can be at an arbitrary
		/// position in the array, this position is stored in the node.
		std::vector<Node> tree;
	};

	inline Expression::Node::Node(float v) :
		value(v),
		flags(0) 
	{ }

	inline Expression::Node::Node(char name) :
		variableName(name),
		flags(0) 
	{
		flags |= SymbolicValue;
	}

	inline Expression::Node::Node(Expression::Operator op) :
		flags(0) 
	{
		flags |= Intermediate;
		flags |= static_cast<unsigned char>(op) << 3;
	}

	inline Expression::Node::Node(int leftIndex, Expression::Operator op) :
		leftIndex(leftIndex),
		flags(0)
	{
		flags |= Intermediate;
		flags |= TwoOperands;
		flags |= static_cast<unsigned char>(op) << 3;
	}

	inline bool Expression::Node::isLeaf() const {
		return (flags & 1) == 0;
	}

	inline uint8_t Expression::Node::getNumberOfArguments() const {
		assert(!isLeaf());
		return (flags & TwoOperands) != 0 ? 2 : 1;
	}

	inline bool Expression::Node::isSymbolic() const {
		return (flags & SymbolicValue) != 0;
	}

	inline Expression::Operator Expression::Node::getOperator() const {
		assert(!isLeaf());
		return Expression::Operator(flags >> 3);
	}

	inline int Expression::Node::getLeftIndex() const {
		assert(!isLeaf());
		return leftIndex;
	}

	inline float Expression::Node::getValue() const {
		assert(isLeaf());
		return value;
	}

	inline char Expression::Node::getName() const {
		assert(isSymbolic());
		return variableName;
	}
}