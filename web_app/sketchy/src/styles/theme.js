import { extendTheme } from "@chakra-ui/react";
import { Button } from "./components/buttonStyles";

export const theme = extendTheme({
  colors: {
    backgroundColor: "#1A365D",
    darkGray: "#A3A8B0",
    lightGray: "#edf4ff",
    gray: "#F7FAFC",
    white: "#FFFFFF",
  },
  components: {
    Button,
  },
});
