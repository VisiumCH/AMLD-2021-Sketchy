import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  ChakraProvider,
  Grid,
  GridItem,
  Text,
  Heading,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from "@chakra-ui/react";
import {
  darkGray,
  textColor,
  backgroundColor,
  buttonHeight,
  progress,
} from "./constants";
import { PageDrawer } from "./drawer.js";
import { BiRefresh } from "react-icons/bi";

function ImagePerformance() {
  const buttonWidth = "230px";
  const [imageType, setImageType] = useState("inference");
  const [images, setImages] = useState([]);
  const [epochImage, setEpochImage] = useState(progress);
  const [epoch, setEpoch] = useState(0);
  const [reload, setReload] = useState(true);

  useEffect(() => {
    setImages([]);
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
    if (reload === true) {
      setReload(false);
      getRandomImage();
    }
  }, [reload]);

  function showImage() {
    if (images.length === 0) {
      return (
        <>
          <Box
            bg={backgroundColor}
            align="center"
            pr={10}
            pl={10}
            pb={10}
          ></Box>
          {progress}
          <Text fontSize="l" color={textColor} align="center">
            Loading random image for all epochs.
          </Text>
        </>
      );
    } else {
      return (
        <>
          <Box bg={backgroundColor} align="center" pr={10} pl={10} pb={5}>
            {epochImage};
          </Box>
          <Button
            leftIcon={<BiRefresh />}
            border="2px"
            borderColor={darkGray}
            variant="solid"
            size="lg"
            onClick={() => {
              setReload(true);
            }}
          >
            Load another image
          </Button>
        </>
      );
    }
  }

  function selectOption(option, text) {
    return (
      <>
        <Button
          color={backgroundColor}
          border="2px"
          borderColor={darkGray}
          variant="solid"
          size="lg"
          height={buttonHeight}
          width={buttonWidth}
          onClick={() => {
            setImageType(option);
            setReload(true);
          }}
        >
          {text}
        </Button>
      </>
    );
  }

  return (
    <ChakraProvider>
      <Box bg={backgroundColor} align="center">
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
          h="91vh"
          w="98vw"
          gap={4}
          align="center"
          templateRows="repeat(8, 1fr)"
          templateColumns="repeat(3, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={1}>
            {selectOption("inference", "See Inference")}
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            {selectOption("attention_sketch", "See Attention on Sketch")}
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}>
            {selectOption("attention_image", "See Attention on Image")}
          </GridItem>
          <GridItem rowSpan={1} colSpan={3}>
            <Box bg={backgroundColor} align="center" pr={40} pl={40}>
              <Text fontSize="xl" color={textColor} align="center">
                Epoch number: {epoch}
              </Text>
              <Slider
                aria-label="slider-ex-2"
                colorScheme="red"
                defaultValue={images.length - 1}
                min={0}
                max={images.length - 1}
                step={1}
                onChangeEnd={(val) => {
                  setEpoch(val);
                  setEpochImage(images[val]);
                }}
              >
                <SliderTrack>
                  <Box position="relative" right={10} />
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
            </Box>
          </GridItem>

          <GridItem rowSpan={5} colSpan={3}>
            {showImage()}
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
