import React, { useState, useCallback, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { Box, ChakraProvider, Button, Text, Heading, VStack, Grid, GridItem, Stack } from '@chakra-ui/react'

const gray = "#F7FAFC"
const darkGray = "#A3A8B0"
const textColor = "#FFFFFF"
const backgroundColor = "#1A365D"
const buttonHeight = "48px"
const buttonWidth = "180px"

const colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000']

function Embeddings() {
    const { state } = useLocation()
    const [isSending, setIsSending] = useState(false)
    const [result, setResult] = useState({})
    const [nbDimensions, setNbDimensions] = useState(3)
    let traces = []

    useEffect(() => {
    }, [state])

    async function getEmbeddings() {
        // Send to back end
        const response = await fetch('/get_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "nb_dim": nbDimensions
            })
        })

        if (response.ok) {
            const res = await response.json()
            setResult(res)
        }
    }

    async function addSketch() {
        // Send to back end
        const response = await fetch('/get_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "sketch": state,
                "nb_dim": nbDimensions
            })
        })
        if (response.ok) {
            const res = await response.json()
            setResult(res)
        }
    }


    const sendRequest = useCallback(async (my_function) => {
        // don't send again while we are sending
        if (isSending) return
        // update state
        setIsSending(true)

        // set images and labels
        my_function()

        // once the request is sent, update state again
        setIsSending(false)

    }, [isSending]) // update the callback if the state changes

    function fillTraces() {
        let marker_size = 4
        let i = 0
        for (let key in result) {
            if (key === "My Custom Sketch") {
                marker_size = 8
            } else {
                marker_size = 4
            }
            let trace = {}
            if (nbDimensions === 3) {
                trace = {
                    x: result[key]['x'],
                    y: result[key]['y'],
                    z: result[key]['z'],
                    name: key,
                    type: 'scatter3d',
                    mode: 'markers',
                    marker: {
                        color: colors[i],
                        size: marker_size
                    },
                    hoverinfo: "name",
                    hovermode: "closest"
                }
            } else {
                trace = {
                    x: result[key]['x'],
                    y: result[key]['y'],
                    name: key,
                    type: 'scatter',
                    mode: 'markers',
                    marker: {
                        color: colors[i],
                        size: marker_size * 2
                    },
                    hoverinfo: "name"
                }
            }
            traces.push(trace)
            i = i + 1
        }
    }
    fillTraces()

    return (
        <ChakraProvider >
            <Box bg={backgroundColor}>
                <VStack
                    spacing={4}
                    align="center">
                    <Heading fontSize="4xl" color={textColor} align="center">
                        AMLD 2021 Visium's Sketchy App
                </Heading>
                    <Text fontSize="2xl" color={textColor} align="center">
                        Embeddings: Images and Sketches in latent space
                </Text>
                </VStack>

                <Grid h="90vh" w="98vw" gap={4} align="center"
                    templateRows="repeat(1, 1fr)" templateColumns="repeat(7, 1fr)">
                    <GridItem rowSpan={1} colSpan={1}  >
                        <VStack spacing={3} direction="row" align="center">
                            <Text fontSize="2xl" color={textColor} align="center">
                                Select a dimension for the graph
                        </Text>
                            <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                                setNbDimensions(2)
                            }}>
                                2D
                    </Button>
                            <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                                setNbDimensions(3)
                            }}>
                                3D
                    </Button>
                            <Text fontSize="2xl" color={textColor} align="center">
                                --------------------------
                        </Text>
                            <Text fontSize="2xl" color={textColor} align="center">
                                Load the graph
                        </Text>
                            <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                                sendRequest(getEmbeddings)
                            }}>
                                Load Graph
                    </Button>
                            <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                                sendRequest(addSketch)
                            }}>
                                Add My Sketch
                    </Button>
                            <Text fontSize="2xl" color={textColor} align="center">
                                --------------------------
                        </Text>
                            <Text fontSize="2xl" color={textColor} align="center">
                                Go back to drawing
                        </Text>
                            <Link to="/drawing" className="drawing_link">
                                <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth}> Back to Drawing</Button>
                            </Link>

                        </VStack>
                    </GridItem>
                    <GridItem rowSpan={1} colSpan={6}  >
                        <Plot
                            data={traces}
                            layout={{
                                width: 1300,
                                height: 700,
                                showlegend: true,
                                margin: {
                                    l: 0,
                                    r: 0,
                                    b: 0,
                                    t: 0
                                },
                                legend: {
                                    title: {
                                        text: 'Categories',
                                        font: {
                                            size: 20,
                                            color: backgroundColor
                                        },
                                    },
                                    font: {
                                        size: 16,
                                        color: backgroundColor
                                    },
                                    orientation: 'v',
                                    itemsizing: "constant",
                                    x: 0.9,
                                    y: 0.5
                                },
                                font: {
                                    color: backgroundColor
                                },
                                paper_bgcolor: gray
                            }}
                        />
                    </GridItem>

                    <Text fontSize="xs" color={textColor} align="center">
                    </Text>
                </Grid>
            </Box>
        </ChakraProvider >
    )
}

export default Embeddings
