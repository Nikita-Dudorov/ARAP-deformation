#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/lscm.h>
#include "ARAPMethod.cpp"



MatrixXd V;
MatrixXi F;
MatrixXd C;

ARAPMethod deformer;
SparseLU<SparseMatrix<double>> ARAPMethod::solver = SparseLU<SparseMatrix<double>>();
//LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> ARAPMethod::solver = LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>>();
std::vector<int> base_vertices,drag_vertices; //stable points and handle points

void Rotation(MatrixXd &V, RowVector3d u, double theta) 
{
    double w = cos(theta/2);
    RowVector3d im = (u / u.norm()) * sin(theta/2);
    Quaterniond q = Quaterniond(w, im[0],im[1],im[2]); 
    for(int i = 0; i < V.rows(); ++i)
    {
        RowVector3d point = V.row(i);
        Quaterniond rotated_point = q*Quaterniond(0, point[0],point[1],point[2])*q.inverse();
        V.row(i) = RowVector3d(rotated_point.x(), rotated_point.y(), rotated_point.z());
    }
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) // Mode selection
{
  // For rotation of handle points around their geometric center
    if (key == 'Q') // Rotate the handle points counter clockwise
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d rotation_axis(0,0,1);
      double angle = M_PI/16;
      Vector3d center = positions.colwise().mean();
      positions = positions.rowwise() - center.transpose();
      Rotation(positions,rotation_axis,angle);
      positions = positions.rowwise() + center.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if(key == 'E') // Rotate the handle points clockwise
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d rotation_axis(0,0,1);
      double angle = -M_PI/16;
      Vector3d center = positions.colwise().mean();
      positions = positions.rowwise() - center.transpose();
      Rotation(positions,rotation_axis,angle);
      positions = positions.rowwise() + center.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    // For translation deformation 
    else if (key == 'S')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(0,-1,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'W')
    {
      MatrixXd positions(drag_vertices.size(), 3);
        for(int i = 0; i < drag_vertices.size(); i++)
        {
          positions.row(i) = V.row(drag_vertices.at(i));
        } 
      Vector3d shift(0,1,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'A')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(-1,0,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'D')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(1,0,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

  return false;
}

//to define control points via hard code
void define_base(std::vector<int> &base_vertices, std::vector<int> &drag_vertices)
{
  //define bottom and top as control points for the mesh 'bar2.off'
  int N = 5*5; //depends on mesh
  int n = 5*4 - 4;

  for(int v=0;v<N;++v)
  {
    base_vertices.push_back(v);
    drag_vertices.push_back(v+N);
  }
  for(int v=2*N;v<2*N+n;++v)
  {
    base_vertices.push_back(v);
  }
  for(int v = 0; v < n; ++v)
  {
    drag_vertices.push_back(V.rows()-1-v);
  }
}

void scale_mesh(MatrixXd &mesh)
{
  double scale_factor = 10/(V.colwise().maxCoeff() - V.colwise().minCoeff()).mean();
  Vector3d center = mesh.colwise().mean();
  mesh = mesh.rowwise() - center.transpose();
  mesh = mesh*scale_factor;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  if(argc<2) {
		std::cout << "Error: input file required (.OFF)" << std::endl;
		return 0;
	}
	std::cout << "reading input file: " << argv[1] << std::endl;

  // Load a mesh in OFF format
  igl::readOFF(argv[1], V, F);
  //to scale mesh so that we can use the same shift values for all meshes
  scale_mesh(V);

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.callback_key_down = &key_down;
 // viewer.callback_mouse_down = &mouse_down;

  // Disable wireframe
  viewer.data().show_lines = true;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // Define not mouvable vertices and vertices moved by user  
  define_base(base_vertices,drag_vertices);

  // create deformer
  deformer = ARAPMethod(V, F, base_vertices, drag_vertices);

  // visualisation of mouvable and dragged vertices
  C = MatrixXd (V.rows(),3);
  C.setConstant(255);
  for(int i=0; i<base_vertices.size();++i)
      C.row(base_vertices[i])=Vector3d(255,0,0);
  for(int i=0; i<drag_vertices.size();++i)
      C.row(drag_vertices[i])=Vector3d(255,255,0);
  // Assign per-vertex colors
	viewer.data().set_colors(C);

  // Launch the viewer
  viewer.launch();
}