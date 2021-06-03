import React, { useState, useCallback, useEffect } from "react";
import {
  Box,
  Button,
  ChakraProvider,
  Grid,
  GridItem,
  Text,
  Heading,
} from "@chakra-ui/react";
import {
  darkGray,
  textColor,
  backgroundColor,
  buttonHeight,
} from "./constants";
import { PageDrawer } from "./drawer.js";

function ImagePerformance() {
  const buttonWidth = "230px";
  const [imageType, setImageType] = useState("inference");
  const [images, setImages] = useState([]);

  useEffect(() => {
    async function getRandomImage() {
      // Send to back end
      const response = await fetch("/image_perf", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_type: imageType,
        }),
      });

      if (response.ok) {
        const res = await response.json();
        const nb_images = res["length"];
        let i;
        let img = "";
        for (i = 0; i < nb_images; i++) {
          img = res[String(i)].split("'")[1];
          let image = <img src={`data:image/jpeg;base64,${img}`} alt="img" />;
          setImages((images) => [...images, image]);
        }
      }
    }
    getRandomImage();
  }, [imageType]);

  return (
    <ChakraProvider>
      <Box bg={backgroundColor} align="center">
        <Heading fontSize="4xl" color={textColor} align="center">
          AMLD 2021 Visium's Sketchy App
        </Heading>
        <Text fontSize="xs" color={textColor} align="center">
          --------------------------------------------------------
        </Text>
        <Grid
          h="93vh"
          w="98vw"
          gap={4}
          align="center"
          templateRows="repeat(8, 1fr)"
          templateColumns="repeat(3, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={1}>
            <Button
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                setImageType("inference");
              }}
            >
              See Inference
            </Button>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Button
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                setImageType("attention_sketch");
              }}
            >
              See Attention on Sketch
            </Button>
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            <Button
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                setImageType("attention_image");
              }}
            >
              See Attention on Images
            </Button>
          </GridItem>
          <GridItem rowSpan={3} colSpan={3}>
            <Text fontSize="xs" color={textColor} align="center">
              --------------------------------------------------------
            </Text>
          </GridItem>
          <GridItem rowSpan={3} colSpan={3}>
            <Text fontSize="xs" color={textColor} align="center">
              --------------------------------------------------------
            </Text>
          </GridItem>
          <GridItem rowSpan={1} colSpan={3}>
            {PageDrawer()}
          </GridItem>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default ImagePerformance;
