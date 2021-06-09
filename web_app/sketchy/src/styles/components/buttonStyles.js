export const Button = {
  // style object for base or default style
  baseStyle: {
    // fontWeight: "bold",
  },
  // styles for different sizes ("sm", "md", "lg")
  sizes: {},
  // styles for different visual variants ("outline", "solid")
  variants: {
    primary: {
      bg: "lightGray",
      color: "backgroundColor",

      height: "48px",
      width: "180px",

      border: "2px",
      borderColor: "darkGray",
    },
    secondary: {
      bg: "lightGray",
      color: "backgroundColor",

      height: "48px",
      width: "210px",

      border: "2px",
    },
  },
  // default values for `size` and `variant`
  defaultProps: {
    variant: "outline",
    size: "lg",
  },
};
