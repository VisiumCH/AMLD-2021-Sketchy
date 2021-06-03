import React from "react";
import { Link } from "react-router-dom";
import {
  Text,
  Button,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  useDisclosure,
  VStack,
} from "@chakra-ui/react";
import {
  backgroundColor,
  darkGray,
  buttonHeight,
  buttonWidth,
} from "./constants";
import {
  BiImage,
  BiImages,
  BiPencil,
  BiShapePolygon,
  BiFoodMenu,
} from "react-icons/bi";
import { AiOutlineLineChart } from "react-icons/ai";

export function PageDrawer(svg) {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const btnRef = React.useRef();

  return (
    <>
      <Button
        leftIcon={<BiFoodMenu />}
        color={backgroundColor}
        borderColor={darkGray}
        height={buttonHeight}
        width={buttonWidth}
        border="2px"
        variant="solid"
        size="lg"
        onClick={onOpen}
      >
        Change Page
      </Button>
      <Drawer
        isOpen={isOpen}
        placement="right"
        onClose={onClose}
        finalFocusRef={btnRef}
      >
        <DrawerOverlay />
        <DrawerContent>
          <DrawerCloseButton />
          <DrawerHeader>Sketchy App</DrawerHeader>

          <DrawerBody>
            <VStack spacing="24px">
              <Text fontSize="xl" color={backgroundColor} align="center">
                Interact with data
              </Text>
              <Link to="/" className="explore_link">
                <Button
                  leftIcon={<BiImage />}
                  color={backgroundColor}
                  borderColor={darkGray}
                  height={buttonHeight}
                  width={buttonWidth}
                  border="2px"
                  variant="solid"
                  size="lg"
                >
                  {" "}
                  Dataset
                </Button>
              </Link>
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
              <Link
                to={{
                  pathname: "/embeddings",
                  state: svg,
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
            </VStack>
          </DrawerBody>
          <DrawerBody>
            <VStack spacing="24px">
              <Text fontSize="xl" color={backgroundColor} align="center">
                See model performance
              </Text>
              <Link
                to={{
                  pathname: "/scalar_perf",
                }}
              >
                <Button
                  leftIcon={<AiOutlineLineChart />}
                  color={backgroundColor}
                  borderColor={darkGray}
                  height={buttonHeight}
                  width="210px"
                  border="2px"
                  variant="solid"
                  size="lg"
                >
                  {" "}
                  Scalar Performance
                </Button>
              </Link>
              <Link
                to={{
                  pathname: "/image_perf",
                }}
              >
                <Button
                  leftIcon={<BiImages />}
                  color={backgroundColor}
                  borderColor={darkGray}
                  height={buttonHeight}
                  width="210px"
                  border="2px"
                  variant="solid"
                  size="lg"
                >
                  {" "}
                  Image Performance
                </Button>
              </Link>
            </VStack>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </>
  );
}
