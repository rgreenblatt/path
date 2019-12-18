from sympy import symbols, GreaterThan, simplify, solve


def main():
    s_0, s_1, s_2 = symbols('s_0 s_1 s_2')
    x_b_0, x_b_1, x_b_2 = symbols('x_b_0 x_b_1 x_b_2')
    y_b_0, y_b_1, y_b_2 = symbols('y_b_0 y_b_1 y_b_2')
    x, y = symbols('x y')

    p_0_x = s_0 * x + x_b_0
    p_1_x = s_1 * x + x_b_1
    p_2_x = s_2 * x + x_b_2

    p_0_y = s_0 * y + y_b_0
    p_1_y = s_1 * y + y_b_1
    p_2_y = s_2 * y + y_b_2

    expr = GreaterThan(
        (p_1_x - p_0_x) * (p_2_y - p_1_y) - (p_1_y - p_0_y) * (p_2_x - p_1_x),
        0)

    print(simplify(solve(expr, y)))

if __name__ == "__main__":
    main()
