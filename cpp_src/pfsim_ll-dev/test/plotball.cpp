#include <TCanvas.h>
#include <TH3F.h>
#include <TH2F.h>
#include <TStyle.h>
#include <src/mesh3d.hpp>
#include <src/sim/findpos.hpp>

#ifndef NUM_ITER
#define NUM_ITER 100
#endif

#ifndef STEP_SIZE
#define STEP_SIZE 2
#endif

template <typename R, typename S, typename T>
void plotball(mesh::thd::Mesh3d<R>& nop, mesh::thd::Mesh3d<T>& val, mesh::thd::Mesh3d<S>& id, const char outfilename[]) {
	auto c06 = new TCanvas("c06","c06",600,400);
	gStyle->SetOptStat(kFALSE);
	auto h3 = new TH3F("h3","distrib. of id(op)=1",nop.xsize(),0,nop.xsize(),nop.ysize(),0,nop.ysize(),nop.zsize(),0,nop.zsize());
   
	// loop over all dimensions and fill histogram
	for (std::size_t i=0; i<nop.xsize(); ++i) {
		for (std::size_t j=0; j<nop.ysize(); ++j) {
			for (std::size_t l=0; l<nop.ysize(); ++l) {
				
				// find where we have the value for id 1
				std::size_t pos = sim::findpos(nop(i,j,l), &id(i,j,l,0), (unsigned short) 1);
				if (pos == nop(i,j,l)) continue;
				
				// fill histogram
				h3->Fill(i,j,l,val(i,j,l,pos));
			}
		}
	}
	

	h3->Draw("BOX1");
	c06->SaveAs(outfilename);
	
	delete c06;
	delete h3;
}

template <typename R, typename S, typename T>
void plotslicesq(std::size_t sliceposz,
				 mesh::thd::Mesh3d<R>& nop, mesh::thd::Mesh3d<T>& val, mesh::thd::Mesh3d<S>& id, 
				 const char outfilename[]) {
	
	// create object to store sumsq to
	mesh::thd::Mesh3d<T> opsq(val.xsize(), val.ysize(), val.zsize());
	
	mesh::thd::sumsq(nop, val, opsq);
	
	
	auto c07 = new TCanvas("c07","c07",600,400);
	gStyle->SetOptStat(kFALSE);
	auto h2 = new TH2F("h2",outfilename,nop.xsize(),0,nop.xsize(),nop.ysize(),0,nop.ysize());
   
	// loop over all dimensions and fill histogram
	for (std::size_t i=0; i<nop.xsize(); ++i) {
		for (std::size_t j=0; j<nop.ysize(); ++j) {
			
			// fill histogram
			h2->Fill(i,j,opsq(i,j,sliceposz));
		}
	}
	
	h2->SetMaximum(1);
	h2->SetMinimum(0.5);
	h2->Draw("COLZ");
	c07->SaveAs(outfilename);
	
	delete c07;
	delete h2;
}

int main() {
	
	// define system size
	const std::size_t m = 41;
	const std::size_t n = 41;
	const std::size_t k = 41;
	
	// initialize test case
	mesh::thd::Mesh3d<unsigned short> nopA(m,n,k);
	mesh::thd::Mesh3d<float> valA(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> idA(m,n,k,0);
	
	for (std::size_t i=0; i<=NUM_ITER; i+=STEP_SIZE) {
		
		// reset nop
		nopA.memset(0);
	
		// read from file
		char filename[64];
		std::sprintf(filename,"./../workdir/3dballtestcase/partition_step%03ld",i);
		auto linesread = mesh::thd::init_from_file(filename, nopA, valA, idA);
		// check for success
		if (linesread==-1 || linesread==0) {
			std::printf("data file input error: %d, aborting\n",linesread);
			return (2);
		}
		std::printf("lines read: %d\n", linesread);
		
		
		char imagefilename[64];
		std::sprintf(imagefilename,"./images/sliceimage%03ld.png",i);
		plotslicesq(k/2, nopA, valA, idA, imagefilename);
	}
	
	
}
