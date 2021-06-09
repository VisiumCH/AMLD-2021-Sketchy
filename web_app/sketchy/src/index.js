import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter as Router } from "react-router-dom";
import { Box, ChakraProvider } from "@chakra-ui/react";
import { theme } from "./styles/theme";
import App from "./App";

ReactDOM.render(
  <React.StrictMode>
    <Router>
      <ChakraProvider resetCSS theme={theme}>
        <Box bg="backgroundColor" h="100%" w="100%" align="center">
          <App />
        </Box>
      </ChakraProvider>
    </Router>
  </React.StrictMode>,
  document.getElementById("root")
);
