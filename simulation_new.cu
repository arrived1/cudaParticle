#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include <utility>
#include <cmath>
#include <string>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <cuda.h>
#include <helper_functions.h>
#include <math_functions.h>

//Constants
const unsigned int window_width = 1024;
const unsigned int window_height = 768;

const unsigned int mesh_width = 1024;
const unsigned int mesh_height = 1024;

//Mouse controls
int mouse_x, mouse_y;
int buttons = 0;
float translate_z = -3.0;

//VBO variables
GLuint vbo;
void *d_vbo_buffer = NULL;

long long unsigned step = 0;

//Device pointers
float4 *d_vel, *d_initPos;

//FPS
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
StopWatchInterface *timer = NULL;


void keyboard(unsigned char key, int, int);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void initGL(int argc, char **argv);
void display(void);
void computeFPS();


union Color
{
    float c;
    uchar4 components;
};

__device__ float4 nextPos[1024*1024], nextVel[1024*1024];

__device__ unsigned idx(unsigned x, unsigned y)
{
    return x * 1024 + y;
}

__global__ void initialize_kernel(float4* pos, unsigned int width, unsigned int height, float4* vel, float4* initPos)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //Set the initial color
    Color tmp;
    tmp.components = make_uchar4(0,255,255,1);

    //Set initial position, color and velocity
    unsigned i = idx(x, y);
    pos[i] = make_float4(initPos[i].x, 0.f, initPos[i].y, tmp.c);
    vel[i] = make_float4(15.0, 0.0, 0.0, 1.0f);
}

__global__ void particles_kernel(float4* pos, unsigned int width, unsigned int height, float4* vel)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned i = idx(x, y);

    float dt = 0.0005f;

    // movement
    pos[i].x = pos[i].x + vel[i].x * dt;
    pos[i].y = pos[i].y + vel[i].y * dt;
    pos[i].z = pos[i].z + vel[i].z * dt;


    // collision
    Color tmp;
    tmp.components = make_uchar4(0,1,0,1);

    float radius = 0.0002f;
    float radius_square = radius * radius;
    unsigned max = 1024 * 1024;
    
    for(unsigned j = 0; j < max; j++)
    {
        if (i == j)
            continue;

        float4 *p1 = &pos[i];
        float4 *p2 = &pos[j];
        
        float dist_square = (p2->x - p1->x) * (p2->x - p1->x) +
                            (p2->y - p1->y) * (p2->y - p1->y);

        // detect collision
        if(dist_square <= radius_square)
        {
            nextPos[i].x = i;
            // update pos
            //nextPos[i] = *p1;
            //nextPos[i].w = tmp.c;

            // update vel
            // nextVel[i].x = vel[i].z * -1.f;
            // nextVel[i].y = vel[i].z * -1.f;
            // nextVel[i].z = vel[i].z * -1.f;
        }
    }
}

__global__ void particle_kernel_update(float4* pos, float4* vel)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned i = idx(x, y);

    pos[i] = nextPos[i];
    vel[i] = nextVel[i];
}

void particles(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *d_pos;
    cudaGLMapBufferObject((void**)&d_pos, vbo);
    
    //Run the particles kernel
    dim3 block(8, 8, 1); // 8 x 8
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1); //128 x 128
    particles_kernel<<< grid, block>>>(d_pos, mesh_width, mesh_height, d_vel);
    //particle_kernel_update<<< grid, block>>>(d_pos);
    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}

void initialize(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *d_pos;
    cudaGLMapBufferObject((void**)&d_pos, vbo);

    //Run the initialization kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    initialize_kernel<<< grid, block>>>(d_pos, mesh_width, mesh_height, d_vel, d_initPos);

    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}

void createVBO(GLuint* vbo)
{
    //Create vertex buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    //Initialize VBO
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //Register VBO with CUDA
    cudaGLRegisterBufferObject(*vbo);
}


int main(int argc, char** argv)
{
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    initGL(argc, argv);

    //Create VBO
    createVBO(&vbo);

    int counter = 0;
    float dist = 0.001f;
    float x_dim = -4.f;
    float4 * h_initPos = (float4*)malloc(mesh_width * mesh_height * sizeof(float4));  
    for (float i = x_dim; i < x_dim + mesh_height * dist; i += dist)
        for (float j = 0.f; j < mesh_width * dist; j += dist)
            h_initPos[counter++] = make_float4(float(i), float(j), 0.f, 0.f);

    //CUDA allocation and copying
    cudaMalloc(&d_vel, mesh_width * mesh_height * sizeof(float4));

    cudaMalloc(&d_initPos, mesh_width * mesh_height * sizeof(float4));
    cudaMemcpy(d_initPos, h_initPos, mesh_height * mesh_width * sizeof(float4), cudaMemcpyHostToDevice);

    initialize(vbo);

    glutMainLoop();

    //Free CUDA variables
    cudaFree(d_vel);
    cudaFree(d_initPos);

    return 0;
}



/////////////////////////////////////// HELPERS /////////////////////////
#define MAX(a,b) ((a > b) ? a : b)

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "simulation (%s): %3.1f fps","", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}


/////////////////////////////////////// OPENGL //////////////////////////

void display(void)
{
    sdkStartTimer(&timer);

    //Process particles using CUDA kernel
    particles(vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //View matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(90.0, 1.0, 0.0, 0.0);

    //Render from VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 16, 0);
    glColorPointer(4,GL_UNSIGNED_BYTE,16,(GLvoid*)12);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    std::cout << step++ << std::endl;

    if(step > 10)
        exit(69);

    sdkStopTimer(&timer);
    computeFPS();
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);

    //Setup window
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Million particles");

    //Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    //GLEW initialization
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(0);
    }

    //Clear
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    //Viewport
    glViewport(0, 0, window_width, window_height);

    //Projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
}

void keyboard(unsigned char key, int, int)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    case('a'):
        if (buttons!=10)
            buttons=10;
        else
            buttons=0;
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        buttons = 0;
    }

    mouse_x = x;
    mouse_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    //float dx = x - mouse_x;
    float dy = y - mouse_y;

    if (buttons & 4)
        translate_z += dy * 0.01;

    mouse_x = x;
    mouse_y = y;
}