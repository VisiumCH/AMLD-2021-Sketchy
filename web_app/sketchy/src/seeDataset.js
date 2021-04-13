import React, { useState, useCallback, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Box,
  ChakraProvider,
  Text,
  Heading,
  Select,
  Button,
  Grid,
  GridItem,
  FormControl,
  HStack,
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
  nb_to_show,
} from "./constants";
import { BiPencil, BiShapePolygon, BiRefresh } from "react-icons/bi";

function Dataset() {
  const [isSending, setIsSending] = useState(false);
  const [currentCategory, setCurrentCategory] = useState("pineapple");
  const [images, setImages] = useState([]);
  const [sketches, setSketches] = useState([]);
  const categoriesOptions = categories.map((category) => (
    <option>{category}</option>
  ));

  async function getImages(currentCategory) {
    // Send to back end
    const response = await fetch("/get_dataset_images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ category: currentCategory }),
    });

    if (response.ok) {
      let tmpImages = [];
      let tmpSketches = [];
      let tmpImage = "";
      let tmpSketch = "";
      const res = await response.json();
      for (let i = 0; i < nb_to_show; i++) {
        tmpImage = res["images_" + i + "_base64"].split("'")[1];
        tmpImages[i] = (
          <img src={`data:image/jpeg;base64,${tmpImage}`} alt="image" />
        );
        tmpSketch = res["sketches_" + i + "_base64"].split("'")[1];
        tmpSketches[i] = (
          <img src={`data:image/jpeg;base64,${tmpSketch}`} alt="sketch" />
        );
      }
      setImages(tmpImages);
      setSketches(tmpSketches);
    }
  }

  const sendRequest = useCallback(
    async (currentCategory) => {
      if (isSending) return;
      setIsSending(true);
      getImages(currentCategory);
      setIsSending(false);
    },
    [isSending]
  );

  useEffect(() => {
    sendRequest(currentCategory);
  }, [sendRequest, currentCategory]);

  const renderImages = images.map((image) => (
    <Box w="50vw" h="30vh">
      {image}
    </Box>
  ));

  const renderSketches = sketches.map((sketch) => (
    <Box w="30vw" h="30vh">
      {sketch}
    </Box>
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
          templateRows="repeat(9, 1fr)"
          templateColumns="repeat(2, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={1} align="right">
            <Text fontSize="2xl" color={textColor}>
              Choose a categories to see some samples:
            </Text>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1} align="left">
            <HStack>
              <form>
                <FormControl
                  id="category"
                  color={backgroundColor}
                  width={formLabelWidth}
                  bg={white}
                >
                  <Select
                    value={currentCategory}
                    color={backgroundColor}
                    onChange={(e) => setCurrentCategory(e.target.value)}
                  >
                    {categoriesOptions}
                  </Select>
                </FormControl>
              </form>
              <Button
                leftIcon={<BiRefresh />}
                border="1px"
                borderColor={darkGray}
                variant="solid"
                size="md"
                onClick={() => {
                  getImages(currentCategory);
                }}
              ></Button>
            </HStack>
          </GridItem>
          <GridItem rowSpan={4} colSpan={2}>
            <HStack align="center">{renderImages}</HStack>
          </GridItem>
          <GridItem rowSpan={4} colSpan={2}>
            <HStack align="center">{renderSketches}</HStack>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Link to="/drawing" className="drawing_link">
              <Button
                leftIcon={<BiPencil />}
                color={backgroundColor}
                borderColor={darkGray}
                height={buttonHeight}
                width={buttonWidth}
                border="2px"
                variant="solid"
                size="lg"
              >
                {" "}
                Draw
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
                leftIcon={<BiShapePolygon />}
                color={backgroundColor}
                borderColor={darkGray}
                height={buttonHeight}
                width={buttonWidth}
                border="2px"
                variant="solid"
                size="lg"
              >
                {" "}
                Embeddings
              </Button>
            </Link>
          </GridItem>
          <GridItem rowSpan={1} colSpan={2} align="right">
            <Text fontSize="xs" color={textColor}>
              .....
            </Text>
          </GridItem>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default Dataset;
