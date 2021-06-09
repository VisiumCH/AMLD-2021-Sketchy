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
      <Button leftIcon={<BiFoodMenu />} variant="primary" onClick={onOpen}>
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
          <DrawerCloseButton color="backgroundColor" />
          <DrawerHeader color="backgroundColor">Sketchy App </DrawerHeader>

          <DrawerBody>
            <VStack spacing="24px">
              <Text fontSize="xl" color="backgroundColor" as="u">
                Interact with data
              </Text>
              <Link to="/" className="explore_link">
                <Button leftIcon={<BiImage />} variant="secondary">
                  {" "}
                  Dataset
                </Button>
              </Link>
              <Link to="/drawing" className="drawing_link">
                <Button leftIcon={<BiPencil />} variant="secondary">
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
                <Button leftIcon={<BiShapePolygon />} variant="secondary">
                  {" "}
                  Embeddings
                </Button>
              </Link>
            </VStack>
          </DrawerBody>
          <DrawerBody>
            <VStack spacing="24px">
              <Text fontSize="xl" color="backgroundColor" as="u">
                See model performance
              </Text>
              <Link
                to={{
                  pathname: "/scalar_perf",
                }}
              >
                <Button leftIcon={<AiOutlineLineChart />} variant="secondary">
                  {" "}
                  Scalar Performance
                </Button>
              </Link>
              <Link
                to={{
                  pathname: "/image_perf",
                }}
              >
                <Button leftIcon={<BiImages />} variant="secondary">
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
