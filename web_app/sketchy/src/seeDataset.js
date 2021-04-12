import { Link } from "react-router-dom";
import {
  Box,
  ChakraProvider,
  Text,
  Heading,
  FormControl,
  FormLabel,
  Select,
  Button,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import {
  textColor,
  backgroundColor,
  darkGray,
  categories,
  buttonHeight,
  buttonWidth,
  formLabelWidth,
  white,
} from "./constants";

function Dataset() {
  const categoriesOptions = categories.map((category) => (
    <option>{category}</option>
  ));

  return (
    <ChakraProvider>
      <Box bg={backgroundColor}>
        <Heading fontSize="4xl" color={textColor} align="center">
          AMLD 2021 Visium's Sketchy App
        </Heading>
        <Text fontSize="xs" color={textColor} align="center">
          --------------------------------------------------------
        </Text>
        <Text fontSize="xs" color={backgroundColor} align="center">
          .
        </Text>
        <Grid
          h="92vh"
          w="98vw"
          gap={4}
          align="center"
          templateRows="repeat(13, 1fr)"
          templateColumns="repeat(2, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={1} align="right">
            <Text fontSize="2xl" color={textColor}>
              Choose a categories to see some samples:
            </Text>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} align="left">
            <FormControl
              id="country"
              color={backgroundColor}
              width={formLabelWidth}
              bg={white}
            >
              <Select placeholder="pineapple" color={backgroundColor}>
                {categoriesOptions}
              </Select>
            </FormControl>
          </GridItem>
          <GridItem rowSpan={10} colSpan={1}>
            <Text fontSize="2xl" color={textColor}>
              ...
            </Text>
          </GridItem>
          <GridItem rowSpan={10} colSpan={1}>
            <Text fontSize="2xl" color={textColor}>
              ...
            </Text>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Link to="/drawing" className="drawing_link">
              <Button
                color={backgroundColor}
                borderColor={darkGray}
                height={buttonHeight}
                width={buttonWidth}
                border="2px"
                variant="solid"
                size="lg"
              >
                {" "}
                Go to Drawing
              </Button>
            </Link>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Link
              to={{
                pathname: "/embeddings",
              }}
            >
              <Button
                color={backgroundColor}
                borderColor={darkGray}
                height={buttonHeight}
                width={buttonWidth}
                border="2px"
                variant="solid"
                size="lg"
              >
                {" "}
                See Embeddings
              </Button>
            </Link>
          </GridItem>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default Dataset;
