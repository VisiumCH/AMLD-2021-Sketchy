import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  IconButton,
  Grid,
  GridItem,
  Text,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from "@chakra-ui/react";
import { progress, heading } from "../ui_utils";
import { PageDrawer } from "../drawer.js";
import { BiRefresh } from "react-icons/bi";
import { CgArrowsHAlt } from "react-icons/cg";

function ImagePerformance() {
  const [imageType, setImageType] = useState("Inference");
  const [images, setImages] = useState([]);
  const [epochImage, setEpochImage] = useState(progress);
  const [epoch, setEpoch] = useState(0);
  const [reload, setReload] = useState(true);

  useEffect(() => {
    setImages([]);
    async function getRandomImage() {
      // Send to back end
      const response = await fetch("/api/image_perf", {
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
          try {
            img = res[String(i)].split("'")[1];
            let image = <img src={`data:image/jpeg;base64,${img}`} alt="img" />;
            setImages((images) => [...images, image]);
          } catch (TypeError) {}
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
          <Box bg="darkBlue" align="center" pr={10} pl={10} pb={10}></Box>
          {progress}
          <Text fontSize="l" color="white" align="center">
            Loading random image for all epochs.
          </Text>
        </>
      );
    } else {
      return (
        <>
          <Box bg="darkBlue" align="center" pr={10} pl={10} pb={5}>
            {epochImage};
          </Box>
          <IconButton
            color="white"
            icon={<BiRefresh />}
            border="2px"
            borderColor="darkGray"
            size="lg"
            borderRadius="30px"
            width="50px"
            onClick={() => {
              setReload(true);
            }}
          ></IconButton>
        </>
      );
    }
  }

  function selectOption(option, text) {
    return (
      <>
        <Button
          variant="primary"
          width="230px"
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
    <>
      {heading}
      <Text fontSize="xs" color="darkBlue" align="center">
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
          {selectOption("Inference", "See Inference")}
        </GridItem>
        <GridItem rowSpan={1} colSpan={1}>
          {selectOption("sketch_attention", "See Attention on Sketch")}
        </GridItem>
        <GridItem rowSpan={1} colSpan={1}>
          {selectOption("image_attention", "See Attention on Image")}
        </GridItem>
        <GridItem rowSpan={1} colSpan={3}>
          <Box bg="darkBlue" align="center" pr={40} pl={40}>
            <Text fontSize="xl" color="white" align="center">
              Epoch number: {epoch}
            </Text>
            <Box bg="darkBlue" align="center" pr={2} pl={2} borderRadius="md">
              <Slider
                aria-label="slider-ex-2"
                colorScheme="red"
                defaultValue={2}
                min={0}
                max={images.length - 1}
                step={1}
                onChangeEnd={(val) => {
                  setEpoch(val);
                  setEpochImage(images[val]);
                }}
              >
                <SliderTrack bg="white">
                  <Box position="relative" right={10} />
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={5}>
                  <Box color="tomato" as={CgArrowsHAlt} />
                </SliderThumb>
              </Slider>
            </Box>
          </Box>
        </GridItem>

        <GridItem rowSpan={5} colSpan={3}>
          {showImage()}
        </GridItem>
        <GridItem rowSpan={1} colSpan={3}>
          {PageDrawer()}
        </GridItem>
      </Grid>
    </>
  );
}

export default ImagePerformance;
