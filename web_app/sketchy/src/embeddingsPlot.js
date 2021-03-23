import React, { useState, useCallback, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { Box, ChakraProvider, Button, Text, Heading, VStack, HStack } from '@chakra-ui/react'

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
            body: JSON.stringify({ "body": "body" })
        })

        if (response.ok) {
            const res = await response.json()
            setResult(res)
        }
    }

    async function addSketch() {
        console.log('Add sketch')
        // Send to back end
        const response = await fetch('/get_sketch_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "sketch": state })
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

    let marker_size = 4
    let i = 0
    for (let key in result) {
        if (key == "My Custom Sketch") {
            marker_size = 8
        } else {
            marker_size = 4
        }
        let trace = {
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
        traces.push(trace)
        i = i + 1
    }



    return (
        <ChakraProvider >
            <Box bg={backgroundColor}>
                <VStack
                    spacing={4}
                    align="center">
                    <Heading fontSize="4xl" color={textColor} align="center">
                        AMLD 2021 Visium's Sketchy App
                </Heading>
                    <Text fontSize="xs" color={textColor} align="center">
                        --------------------------------------------------------
                </Text>
                    <Text fontSize="2xl" color={textColor} align="center">
                        Embeddings: Images and Sketches in latent space
                </Text>
                    <Plot
                        data={traces}
                        layout={{
                            width: 1200,
                            height: 645,
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
                                x: 0.8,
                                y: 0.5
                            },
                            font: {
                                color: backgroundColor
                            },
                            paper_bgcolor: gray
                        }}
                    />
                    <HStack
                        spacing={40}
                        align="center"
                    >
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
                        <Link to="/drawing" className="drawing_link">
                            <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg"> Back to Drawing</Button>
                        </Link>
                    </HStack>
                    <Text fontSize="xs" color={textColor} align="center">

                    </Text>
                </VStack>
            </Box>
        </ChakraProvider >
    )
}

export default Embeddings
