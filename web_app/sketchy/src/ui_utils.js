import { CircularProgress, Heading, Text } from "@chakra-ui/react";

export const progress = (
  <CircularProgress
    isIndeterminate
    color={"darkBlue"}
    size="180px"
    thickness="4px"
  />
);

export const heading = (
  <>
    <Heading fontSize="4xl" color={"white"} align="center">
      AMLD 2021 Visium's Sketchy App
    </Heading>
    <Text fontSize="xs" color={"white"} align="center">
      --------------------------------------------------------
    </Text>
  </>
);
